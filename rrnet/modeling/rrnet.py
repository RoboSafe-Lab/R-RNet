'''
Defined classes:
    class BiTraPNP()
Some utilities are cited from Trajectron++
'''
import sys
import numpy as np
import copy
from collections import defaultdict
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn

from rrnet.modeling.latent_net import CategoricalLatent, kl_q_p
from rrnet.layers.loss import cvae_loss, mutual_inf_mc, prob_loss, attention_loss
import logging
from torch.distributions import Normal
import time


class RRNet(nn.Module):
    def __init__(self, cfg, dataset_name=None): 
        super(RRNet, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.K = self.cfg.K
        self.Latent_sample_num_test = self.cfg.Latent_sample_num_test
        self.Latent_sample_num_train = self.cfg.Latent_sample_num_train
        self.Ra = self.cfg.Ra
        self.randomize = self.cfg.Randomize
        self.random_num= self.cfg.Random_num
        self.mu = 0.0
        self.sigma = 1.0
        self.attention= self.cfg.ATTENTION
        self.param_scheduler = None
        # encoder
        self.box_embed = nn.Sequential(nn.Linear(self.cfg.GLOBAL_INPUT_DIM, self.cfg.INPUT_EMBED_SIZE), # 6,256
                                        nn.ReLU())
        self.traj_embed = nn.Sequential(nn.Linear(self.cfg.GLOBAL_INPUT_DIM, self.cfg.INPUT_EMBED_SIZE), # 6,256
                                        nn.ReLU())
        self.box_encoder = nn.GRU(input_size=self.cfg.INPUT_EMBED_SIZE, # 256
                                hidden_size=self.cfg.ENC_HIDDEN_SIZE, # 256
                                batch_first=True)
        self.box_encoder_traj = nn.GRU(input_size=self.cfg.INPUT_EMBED_SIZE, # 256
                                hidden_size=self.cfg.ENC_HIDDEN_SIZE, # 256
                                batch_first=True)

        #encoder for future trajectory
        # self.gt_goal_encoder = nn.Sequential(nn.Linear(self.cfg.DEC_OUTPUT_DIM, 16),  #2
        #                                         nn.ReLU(),
        #                                         nn.Linear(16, 32),
        #                                         nn.ReLU(),
        #                                         nn.Linear(32, self.cfg.GOAL_HIDDEN_SIZE),
        #                                         nn.ReLU())
        self.node_future_encoder_h = nn.Linear(self.cfg.GLOBAL_INPUT_DIM, 32)   # 6
        self.gt_goal_encoder = nn.GRU(input_size=self.cfg.DEC_OUTPUT_DIM,   # 2
                                        hidden_size=32,
                                        bidirectional=True,
                                        batch_first=True)
        
            
        self.hidden_size = self.cfg.ENC_HIDDEN_SIZE        #256
        self.p_z_x = nn.Sequential(nn.Linear(self.hidden_size,  #256
                                            128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.cfg.LATENT_DIM*2))   #256——>32×2
        # posterior
        self.q_z_xy = nn.Sequential(nn.Linear(self.hidden_size + self.cfg.GOAL_HIDDEN_SIZE,     #256+64(configs/defaults)
                                            128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.cfg.LATENT_DIM*2))      #(256+64)——>32*2

        # goal predictor
        self.goal_decoder = nn.Sequential(nn.Linear(self.hidden_size + self.cfg.LATENT_DIM,     #256+32
                                                    128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, self.cfg.DEC_OUTPUT_DIM))     #(256+32)——>2
        #  add bidirectional predictor
        self.dec_init_hidden_size = self.hidden_size + self.cfg.LATENT_DIM if self.cfg.DEC_WITH_Z else self.hidden_size
                                        # 256
        self.enc_h_to_forward_h = nn.Sequential(nn.Linear( self.dec_init_hidden_size,   #256
                                                      self.cfg.DEC_HIDDEN_SIZE),        #256
                                                nn.ReLU(),
                                                )
        self.traj_dec_input_forward = nn.Sequential(nn.Linear(self.cfg.DEC_HIDDEN_SIZE, #256
                                                              self.cfg.DEC_INPUT_SIZE), #256
                                                    nn.ReLU(),
                                                    )
        self.traj_dec_forward = nn.GRUCell(input_size=self.cfg.DEC_INPUT_SIZE,      #256
                                            hidden_size=self.cfg.DEC_HIDDEN_SIZE)   #256
        
        self.enc_h_to_back_h = nn.Sequential(nn.Linear( self.dec_init_hidden_size,  #256
                                                      self.cfg.DEC_HIDDEN_SIZE),    #256
                                            nn.ReLU(),
                                            )
        
        self.traj_dec_input_backward = nn.Sequential(nn.Linear(self.cfg.DEC_OUTPUT_DIM, # 2 or 4 
                                                                self.cfg.DEC_INPUT_SIZE),   #256
                                                        nn.ReLU(),
                                                        )
        self.traj_dec_backward = nn.GRUCell(input_size=self.cfg.DEC_INPUT_SIZE,     #256
                                            hidden_size=self.cfg.DEC_HIDDEN_SIZE)   #256

        self.traj_output = nn.Linear(self.cfg.DEC_HIDDEN_SIZE * 2, # merged forward and backward 
                                     self.cfg.DEC_OUTPUT_DIM)       #2
        self.confidence_encoder = nn.GRU(input_size=self.cfg.DEC_OUTPUT_DIM,   # 2
                                        hidden_size=32,
                                        bidirectional=True,
                                        batch_first=True)

        # goal predictor
        self.confidence_decoder = nn.Sequential(nn.Linear(64,     #256+32
                                                    32),
                                            nn.ReLU(),
                                            nn.Linear(32, 16),
                                            nn.ReLU(),
                                            nn.Linear(16, 1),
                                            nn.Sigmoid())     #(256+32)——>2

        self.partial_encoder = nn.GRU(6, 256, batch_first=True)
        self.attention_gru= nn.GRU(256, 32, batch_first=True)
        self.goal_attention_decoder = nn.Sequential(nn.Linear(self.hidden_size + self.cfg.LATENT_DIM+32,     #256+32+32
                                                    128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, self.cfg.DEC_OUTPUT_DIM))
        self.cat_dec_h_and_attention=nn.Linear(256+32,
                                     256)
        self.query = nn.Linear(2, 32)
        self.key = nn.Linear(2, 32)
        self.value = nn.Linear(2, 32)
        self.attention_goal_revise=nn.Linear(32,
                                     2)

        
        self.parts_to_freeze_goal1 = nn.ModuleList([
            self.confidence_encoder,
            self.confidence_decoder,
        ])
        self.parts_to_freeze_goal2 = nn.ModuleList([
            self.query,
            self.key,
            self.value,
            self.attention_goal_revise
        ])
        
    def gaussian_latent_net(self, enc_h, cur_state, target=None, z_mode=None, test=False):
        # get mu, sigma
        # 1. sample z from piror
        z_mu_logvar_p = self.p_z_x(enc_h)

        z_mu_p = z_mu_logvar_p[:, :self.cfg.LATENT_DIM]
        z_logvar_p = z_mu_logvar_p[:, self.cfg.LATENT_DIM:]
        if target is not None:
            # 2. sample z from posterior, for training only
            initial_h = self.node_future_encoder_h(cur_state)   # 6-->32
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)   
            _, target_h = self.gt_goal_encoder(target, initial_h)  
            target_h = target_h.permute(1,0,2)  
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])  

            target_h = F.dropout(target_h,
                                p=0.25,
                                training=self.training)

            z_mu_logvar_q = self.q_z_xy(torch.cat([enc_h, target_h], dim=-1))   
            z_mu_q = z_mu_logvar_q[:, :self.cfg.LATENT_DIM]
            z_logvar_q = z_mu_logvar_q[:, self.cfg.LATENT_DIM:]
            Z_mu = z_mu_q
            Z_logvar = z_logvar_q

            # 3. compute KL(q_z_xy||p_z_x)
            KLD = 0.5 * ((z_logvar_p.exp()/z_logvar_q.exp()) + \
                        (z_mu_p - z_mu_q).pow(2)/z_logvar_p.exp() - \
                        1 + \
                        (z_logvar_q-z_logvar_p))
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)
        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = 0.0

        # 4. Draw sample
        #K_samples = torch.randn(enc_h.shape[0], self.K, self.cfg.LATENT_DIM).cuda()     #(128,20,32)
        with torch.set_grad_enabled(False):
            sample_num = self.Latent_sample_num_test if test else self.Latent_sample_num_train
            K_samples = torch.normal(self.mu, self.sigma, size = (enc_h.shape[0], sample_num, self.cfg.LATENT_DIM)).cuda()
        probability = self.reconstructed_probability(K_samples)
        Z_std = torch.exp(0.5 * Z_logvar)   
        Z = Z_mu.unsqueeze(1).repeat(1, sample_num, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, sample_num, 1)    

        if z_mode:
            Z = torch.cat((Z_mu.unsqueeze(1), Z), dim=1)
        return Z, KLD, probability

    def encode_variable_length_seqs(self, original_seqs, lower_indices=None, upper_indices=None, total_length=None,traj=False):
        '''
        take the input_x, pack it to remove NaN, embed, and run GRU
        '''
        
        bs, tf = original_seqs.shape[:2] 
        if lower_indices is None:   #tensor([0, 2, 0, 0,......,0, 1, 0, 0], dtype=torch.int32)
            lower_indices = torch.zeros(bs, dtype=torch.int)
        if upper_indices is None:   # 8-1=7
            upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
        if total_length is None:    #7+1=8
            total_length = max(upper_indices) + 1
        # This is done so that we can just pass in self.prediction_timesteps
        # (which we want to INCLUDE, so this will exclude the next timestep).
        inclusive_break_indices = upper_indices + 1     # 7+1=8
        pad_list = []
        length_per_batch = []
        for i, seq_len in enumerate(inclusive_break_indices): 
            pad_list.append(original_seqs[i, lower_indices[i]:seq_len]) 
            length_per_batch.append(seq_len-lower_indices[i]) #

        # 1. embed and convert back to pad_list
        if traj:
            x = self.traj_embed(torch.cat(pad_list, dim=0))
        else:
            x = self.box_embed(torch.cat(pad_list, dim=0))# sum×6——>sum×256
        pad_list = torch.split(x, length_per_batch) 

        # 2. run temporal
        packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False) 
        if traj:
            packed_output, h_x = self.box_encoder_traj(packed_seqs)
        else:
            packed_output, h_x = self.box_encoder(packed_seqs)

        # pad zeros to the end so that the last non zero value
        output, _ = rnn.pad_packed_sequence(packed_output,     
                                            batch_first=True,
                                            total_length=total_length)


        return output, h_x

    def encoder(self, x, first_history_indices=None, traj=False):
        '''
        x: encoder inputs
        '''
        outputs, _ = self.encode_variable_length_seqs(x,
                                                      lower_indices=first_history_indices,traj=traj)  #[128,8,256]
        outputs = F.dropout(outputs,
                            p=self.cfg.DROPOUT,  # 0.25
                            training=self.training)
        if first_history_indices is not None:
            last_index_per_sequence = -(first_history_indices + 1).to(torch.long)
            return outputs[torch.arange(first_history_indices.shape[0]).to(torch.long), last_index_per_sequence]
        else:
            # if no first_history_indices, all sequences are full length
            return outputs[:, -1, :]

    def forward(self, input_x,
                target_y=None,
                neighbors_st=None,
                adjacency=None,
                z_mode=False,
                cur_pos=None,
                first_history_indices=None):


        '''
        Params:
            input_x: (batch_size, segment_len, dim =2 or 4)
            target_y: (batch_size, pred_len, dim = 2 or 4)
        Returns:
            pred_traj: (batch_size, K, pred_len, 2 or 4)
        '''
        test = True if target_y is None else False


        #input_x:[128,8,6]      target_y:[128,12,2]
        gt_goal = target_y[:, -1] if target_y is not None else None  
        cur_pos = input_x[:, -1, :] if cur_pos is None else cur_pos 
        batch_size, seg_len, _ = input_x.shape
        # 1. encoder
        h_x = self.encoder(input_x, first_history_indices) 

        # 2-3. latent net and goal decoder
        Z, KLD, reconstruct_prob = self.gaussian_latent_net(h_x, input_x[:, -1, :], target_y, test = test, z_mode=False)   
        enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1) #(128,20,256+32)/(128,60,256+32)



        # coarse goal+input_hidden_info

        goal_sampled = self.goal_decoder(enc_h_and_z)  #->2
        dec_h = enc_h_and_z if self.cfg.DEC_WITH_Z else h_x #false
        

        # probability-----------start
        goal_for_prob = goal_sampled.unsqueeze(2)
        prob_input_x = input_x.unsqueeze(1).repeat(1, goal_for_prob.shape[1], 1, 1)[:,:,:,:2] #(128,20,8,2)
        prob_sampled, prob_sampled_sm = self.predict_prob(prob_input_x, goal_for_prob)
        
        # probability-----------finish


        
        if self.attention:
            topK_goal, topK_prob, revised_pred_goal_bias = self.goal_revise_attention(goal_sampled, prob_sampled, self.K)  # (128,20,2) (128,20,1)
            revised_goal = topK_goal + revised_pred_goal_bias
        else:
            revised_goal, _=self.goal_revise(goal_sampled, prob_sampled, self.K)
        # trajectory network


        if test and self.randomize:
            prob_input_x = input_x.unsqueeze(1).repeat(1, revised_goal.shape[1], 1, 1)[:,:,:,:2] #(128,20,8,2)
            revised_goal = self.randomize_goal(prob_input_x, revised_goal, self.Ra, self.random_num)
        pred_traj = self.pred_future_traj(dec_h, revised_goal)  # (128,12,20,2)

        #final goal prob
        final_goal = pred_traj[:,-1].unsqueeze(2)
        prob_input_x = input_x.unsqueeze(1).repeat(1, final_goal.shape[1], 1, 1)[:,:,:,:2] #(128,20,8,2)
        final_prob, final_prob_sm = self.predict_prob(prob_input_x, final_goal)




        # 5. compute loss
        if target_y is not None:
            loss_goal, loss_traj = cvae_loss(goal_sampled,
                                                revised_goal,
                                            pred_traj,
                                            target_y,
                                            best_of_many=self.cfg.BEST_OF_MANY
                                            )
            loss_prob = prob_loss(prob_sampled, goal_sampled, target_y[:, -1, :])
            loss_attention = attention_loss(revised_pred_goal_bias, revised_goal, target_y[:, -1, :])
            loss_dict = {'loss_goal': loss_goal, 'loss_traj': loss_traj,
                        'loss_kld': KLD, 'loss_probability': loss_prob, 'loss_attention':loss_attention}
        else:
            # test
            loss_dict = {}
            




        return goal_sampled, pred_traj, loss_dict, None, None

    def pred_future_traj(self, dec_h, G):   #(128,256)  (128,20,2)
        '''
        use a bidirectional GRU decoder to plan the path.
        Params:
            dec_h: (Batch, hidden_dim) if not using Z in decoding, otherwise (Batch, K, dim) 
            G: (Batch, K, pred_dim)
        Returns:
            backward_outputs: (Batch, T, K, pred_dim)
        '''
        pred_len = self.cfg.PRED_LEN    #12

        K = G.shape[1]
        # 1. run forward
        forward_outputs = []
        forward_h = self.enc_h_to_forward_h(dec_h)  #256
        if len(forward_h.shape) == 2:   #2
            forward_h = forward_h.unsqueeze(1).repeat(1, K, 1)  #(128,20,256)
        forward_h = forward_h.view(-1, forward_h.shape[-1]) #(128×20,256)
        forward_input = self.traj_dec_input_forward(forward_h)
        for t in range(pred_len): # the last step is the goal, no need to predict
            forward_h = self.traj_dec_forward(forward_input, forward_h)
            forward_input = self.traj_dec_input_forward(forward_h)
            forward_outputs.append(forward_h)
        
        forward_outputs = torch.stack(forward_outputs, dim=1)   #(20×128,12,256)
        
        # 2. run backward on all samples
        backward_outputs = []
        backward_h = self.enc_h_to_back_h(dec_h)
        if len(dec_h.shape) == 2:
            backward_h = backward_h.unsqueeze(1).repeat(1, K, 1)
        backward_h = backward_h.view(-1, backward_h.shape[-1])  #(20×128,256)
        backward_input = self.traj_dec_input_backward(G)#torch.cat([G]) (128,20,256)
        backward_input = backward_input.view(-1, backward_input.shape[-1])  #(20×128,256)
        
        for t in range(pred_len-1, -1, -1):
            backward_h = self.traj_dec_backward(backward_input, backward_h)
            output = self.traj_output(torch.cat([backward_h, forward_outputs[:, t]], dim=-1))   #(20×128,256+256->2)
            backward_input = self.traj_dec_input_backward(output)
            backward_outputs.append(output.view(-1, K, output.shape[-1]))   #(128,20,2)
        
        # inverse because this is backward 
        backward_outputs = backward_outputs[::-1]
        backward_outputs = torch.stack(backward_outputs, dim=1) #(128,12,20,2)
        
        return backward_outputs



    def attention_encoder(self, input_x,enc_h_and_z):
        batch_size, seq_len, _ = input_x.size()
        encodings = []
        for i in range(seq_len):
            _, encoding = self.partial_encoder(input_x[:, i:, :])   #[1,128,256]
            encoding = encoding.squeeze(0)  
            encodings.append(encoding)

        encodings = torch.stack(encodings, dim=1)   #[128,8,256]



        query = encodings
        key = encodings
        value = encodings

        # QK^T
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(key.shape[-1], dtype=torch.float32))

        attention_weights = F.softmax(scores, dim=-1)

        attention_output = torch.matmul(attention_weights, value)   # [128, 8, 256]
        _, attention_h=self.attention_gru(attention_output)  # [128,32]
        attention_h = attention_h.squeeze(0)
        enc_h_and_z_and_attention = torch.cat([attention_h.unsqueeze(1).repeat(1, enc_h_and_z.shape[1], 1), enc_h_and_z], dim=-1)  # (128,20,256+32+32)
        attention_goal=self.goal_attention_decoder(enc_h_and_z_and_attention)
        return attention_goal, attention_h

    def reconstructed_probability(self,sample):
        recon_dist = Normal(self.mu, self.sigma)
        p = recon_dist.log_prob(sample).exp().mean(dim=-1)  # [batch_size, K]
        return p

    def goal_revise_attention(self, pred_goal, prob, K):   # (128,20,2) (128,20,1)
        topK_goal, topK_prob = self.goal_revise(pred_goal, prob, K)  # (128,20,2) (128,20,1)
        input = topK_goal*0.3
        Q = self.query(input)
        K = self.key(input)
        V = self.value(input)


        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(32, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)   #[128,20,32]
        revised_pred_goal_bias=self.attention_goal_revise(attention_output)

        return topK_goal, topK_prob, revised_pred_goal_bias

    def goal_revise(self,pred_goal, prob, K):  
        topK_prob, topK_idx = torch.topk(prob, K, dim=1, largest=True)  # [128,20]
        topK_idx = topK_idx.squeeze(-1)
        batch_size = pred_goal.shape[0]

        topK_goal = torch.stack([pred_goal[i, topK_idx[i]] for i in range(batch_size)])  # [128,20,2]

        return topK_goal, topK_prob

    def stat(self, gt_goal, pred_goal, con_traj_sm, reconstruct_prob_sm):  #[128,2],[128,20,2],[128,20,1],[128,20]
        con_traj_sm=con_traj_sm.squeeze(-1)
        prob1=con_traj_sm
        prob2=reconstruct_prob_sm
        batchsize = gt_goal.size(0)

        distances = torch.norm(pred_goal - gt_goal[:, None, :], dim=2)
        min_distance_indices = torch.argmin(distances, dim=1)  # shape: [batchsize]

        max_prob1_indices = torch.argmax(prob1, dim=1)  # shape: [batchsize]
        
        max_prob2_indices = torch.argmax(prob2, dim=1)  # shape: [batchsize]

        prob1_num = torch.sum(min_distance_indices == max_prob1_indices).item()
        prob2_num = torch.sum(min_distance_indices == max_prob2_indices).item()
        prob_stat={"data_num": batchsize, "recon_num": prob2_num, "prob_num": prob1_num}
        return prob_stat

    
    def predict_prob(self, input_x, goal):  #[128,20,8,2], [128,20,1,2]
        prob_input = torch.cat((input_x, goal), dim=2)  #[128,20,9,2]
        prob_input = prob_input.view(-1, prob_input.shape[2],prob_input.shape[3])  # (128×20,9,2)

        _, prob_hn = self.confidence_encoder(prob_input)  #[1,128×20,64]
        prob_hn = prob_hn.permute(1, 0, 2)  
        prob_hn = prob_hn.reshape(-1, prob_hn.shape[1] * prob_hn.shape[2])  #   [128×20,64]

        prob = self.confidence_decoder(prob_hn)    #[128×20,1]
        prob = prob.view(-1, goal.shape[1], 1)  # (128, 20, 1)
        prob_sm = F.softmax(prob, dim=1)
        return prob, prob_sm

    def randomize_goal(self, input_x, goal, alpha, random_num):
        prob, _ = self.predict_prob(input_x, goal.unsqueeze(2))

        num_goals = goal.size(1)
        num_select = random_num  
        _, indices = torch.topk(prob.squeeze(-1), num_select, largest=False, dim=1)

        mask = torch.zeros_like(prob.squeeze(-1), dtype=torch.bool)
        batch_indices = torch.arange(goal.size(0)).unsqueeze(1)
        mask[batch_indices, indices.squeeze(-1)] = True

        selected_goals = goal[mask]


        random_directions = torch.randn_like(selected_goals)  
        random_directions = random_directions / torch.norm(random_directions, dim=-1, keepdim=True) 

        perturbation = alpha * random_directions
        modified_goals = selected_goals + perturbation

        goal[mask] = modified_goals
        return goal