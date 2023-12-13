import torch
from torch import nn
# from eth_ucy.gcl_t import GCL, E_GCL, E_GCL_vel, GCL_rf_vel, DynE_GCL_vel, DynE_GCL_vel_channel,DynE_GCL_vel_channel_clean
import numpy as np
from torch.nn import functional as F
import random


class AuxFormer(nn.Module):
    def __init__(self, 
                in_dim,
                h_dim, 
                past_timestep, 
                future_timestep, 
                mask_ratio, 
                decoder_dim=64, 
                num_heads=8, 
                encoder_depth=3, 
                decoder_depth=1,
                decoder_dim_per_head=32,
                same_head=True,
                range_mask_ratio=False,
                dim_per_head=64,
                mlp_head=False,
                mask_past=False,
                mask_range=20,
                multi_output=False,
                decoder_masking=False,
                pred_all=False,
                mlp_dim=64,
                use_projector=False,
                easier_given_dest=False,
                noise_dev=0.,
                part_noise=False,
                denoise_mode='all',
                part_noise_ratio=0.25,
                add_joint_token=False,
                n_agent=22,
                concat_vel=False,
                concat_acc=False,
                dropout=0.,
                only_recons_past=False,
                two_mask_mode=False,
                add_residual=False,
                denoise=True,
                regular_masking=False,
                multi_same_head=False,
                range_noise_dev=False):
        super(AuxFormer, self).__init__()
        self.all_timesteps = past_timestep + future_timestep
        self.mask_ratio = mask_ratio
        self.range_mask_ratio = range_mask_ratio
        self.pred_num_masked = future_timestep
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))
        self.past_timestep = past_timestep
        self.future_timestep = future_timestep
        self.mask_past = mask_past
        self.mask_range = mask_range
        self.multi_output = multi_output
        self.decoder_masking = decoder_masking
        self.encoder_depth = encoder_depth
        self.patch_embed = nn.Linear(in_dim, h_dim)
        self.decoder_pos_embed = nn.Embedding(self.all_timesteps, decoder_dim)
        self.decoder_agent_embed = nn.Embedding(n_agent, decoder_dim)
        self.only_recons_past = only_recons_past

        self.pred_time = 1
        self.pred_all = pred_all
        self.concat_vel = concat_vel
        self.concat_acc = concat_acc
        if self.concat_vel:
            self.patch_embed = nn.Linear(6, h_dim)
        elif self.concat_acc:
            self.patch_embed = nn.Linear(9, h_dim)
        else:
            self.patch_embed = nn.Linear(3, h_dim)
        self.noise_dev = noise_dev
        self.part_noise = part_noise
        self.part_noise_ratio = part_noise_ratio
        self.denoise_mode = denoise_mode
        self.add_joint_token = add_joint_token
        self.mlp_head = mlp_head
        self.agent_embed = nn.Parameter(torch.randn(1, n_agent, 1, h_dim))
        # self.agent_embed = nn.Parameter(torch.empty(1, n_agent, 1, h_dim)).cuda()
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.all_timesteps, h_dim))
        self.two_mask_mode = two_mask_mode
        self.add_residual = add_residual
        self.regular_masking = regular_masking
        self.multi_same_head = multi_same_head      
        self.range_noise_dev = range_noise_dev

        if self.multi_output:
            self.head = []
            for i in range(self.encoder_depth):
                if mlp_head:
                    self.head.append(MLP(decoder_dim, 3*self.pred_time,hidden_size=(128,)))
                else:
                    self.head.append(nn.Linear(decoder_dim, 3*self.pred_time))
            self.head = nn.ModuleList(self.head)
        else:
            if mlp_head:
                self.head = MLP(decoder_dim, 3*self.pred_time,hidden_size=(decoder_dim,))
            else:
                self.head = nn.Linear(decoder_dim, 3*self.pred_time)
        self.same_head = same_head
        if not same_head:
            if mlp_head:
                self.aux_head = MLP(decoder_dim, 3*self.pred_time,hidden_size=(decoder_dim,))
                # self.aux_head = MLP(decoder_dim, 2*self.pred_time,hidden_size=())
                self.aux_head_2 = MLP(decoder_dim, 3*self.pred_time,hidden_size=(decoder_dim,))
            else:
                self.aux_head = nn.Linear(decoder_dim, 3*self.pred_time)
                self.aux_head_2 = nn.Linear(decoder_dim, 3*self.pred_time)

        self.encoder = []
        self.decoder = []
        for _ in range(self.encoder_depth):
            self.encoder.append(STTrans(self.all_timesteps,h_dim,depth=1,mlp_dim=mlp_dim,num_heads=num_heads,dim_per_head=dim_per_head,dropout=dropout))
            self.decoder.append(STTrans(self.all_timesteps,h_dim,depth=1,mlp_dim=mlp_dim,num_heads=num_heads,dim_per_head=dim_per_head,dropout=dropout))
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self,all_traj):
        # all_traj = all_traj.double()
        B,N,T = all_traj.shape[0],all_traj.shape[1],all_traj.shape[2]
        all_traj = all_traj.view(B,N,T,3)
        start = all_traj[:,:,self.past_timestep-1:self.past_timestep]

        batch_ind = torch.arange(B)[:,None,None].cuda()
        agent_ind = torch.arange(N)[None,:,None].cuda()

        ordinary_mask = torch.zeros((B,N,T)).type_as(all_traj)
        ordinary_mask[:,:,:self.past_timestep] = 1.

        mask_ratio = self.mask_ratio
        if self.range_mask_ratio:
            mask_ratio = random.choice([0.3,0.4,0.5,0.6,0.7])
        
        if self.range_noise_dev:
            noise_dev = np.random.uniform(low=0.1, high=1.0)
        else:
            noise_dev = self.noise_dev

        if self.regular_masking:
            mask = torch.ones((B,N,self.past_timestep)).type_as(all_traj)
            shuffle_indices = torch.rand((B,N,self.past_timestep)).argsort().cuda()
            mask_indices = shuffle_indices[:,:,:int(self.past_timestep*mask_ratio)]
            mask[batch_ind,agent_ind,mask_indices] = 0
            assert torch.sum(mask) == (self.past_timestep-int(self.past_timestep*mask_ratio))*B*N
            mask = mask.view(B,N,-1)
            future_mask = torch.zeros((B,N,self.future_timestep)).type_as(all_traj)
            mask_with_no_future = torch.cat([mask,torch.ones_like(future_mask)],dim=-1)
            mask = torch.cat([mask,future_mask],dim=-1)
        else:
            mask = torch.ones((B,N*self.past_timestep)).type_as(all_traj)
            shuffle_indices = torch.rand((B,N*self.past_timestep)).argsort().cuda()
            mask_indices = shuffle_indices[:,:int(N*self.past_timestep*mask_ratio)]
            batch_ind2 = torch.arange(B)[:,None].cuda()
            mask[batch_ind2,mask_indices] = 0
            assert torch.sum(mask) == (N*self.past_timestep-int(N*self.past_timestep*mask_ratio))*B
            mask = mask.view(B,N,-1)
            future_mask = torch.zeros((B,N,self.future_timestep)).type_as(all_traj)
            mask_with_no_future = torch.cat([mask,torch.ones_like(future_mask)],dim=-1)
            mask = torch.cat([mask,future_mask],dim=-1)

        denoise_mask = torch.zeros((B,N,T)).type_as(all_traj)
        denoise_mask[:,:,:self.past_timestep] = 1.

        noise = torch.from_numpy(np.random.normal(loc=0., scale=noise_dev, size=(B,N,T,3))).type_as(all_traj)
        if self.part_noise:
            noise_mask = torch.rand(B,N,T).cuda()
            noise_mask = (noise_mask < self.part_noise_ratio).type_as(all_traj)
            noise = noise * noise_mask[:,:,:,None].repeat(1,1,1,3)
        all_traj_noise = all_traj + noise

        if self.multi_output:
            out = self.mask_forward(all_traj,ordinary_mask,all_out=True)
        else:
            decoded_tokens = self.mask_forward(all_traj,ordinary_mask)

        decoded_tokens_aux = self.mask_forward(all_traj,mask)
        decoded_tokens_noise = self.mask_forward(all_traj_noise,denoise_mask)

        past_future_indices = torch.arange(T)[None].repeat(B,1).cuda()
        past_future_indices = past_future_indices[:,None,:].repeat(1,N,1)
        past_ind, future_ind = past_future_indices[:,:, :self.past_timestep], past_future_indices[:,:, self.past_timestep:]

        if self.multi_output:
            pred_future_coord_values = []
            for ind in range(len(out)):
                if self.multi_same_head:
                    head_ind = -1
                else:
                    head_ind = ind
                if self.mlp_head:
                    output = self.head[head_ind](out[ind][batch_ind,agent_ind, future_ind, :])
                else:
                    output = self.head[head_ind](out[ind][batch_ind,agent_ind, future_ind, :])
                output = output.view(B,N,-1,3)
                if self.add_residual:
                    output += start
                pred_future_coord_values.append(output)
        else:
            dec_future_tokens = decoded_tokens[batch_ind,agent_ind, future_ind, :]
            pred_future_coord_values = self.head(dec_future_tokens)
            pred_future_coord_values = pred_future_coord_values.view(B,N,-1,3)
            if self.add_residual:
                pred_future_coord_values += start

        dec_mask_tokens = decoded_tokens_aux
        if self.same_head:
            if self.multi_output:
                pred_mask_coord_values = self.head[-1](dec_mask_tokens)
            else:
                pred_mask_coord_values = self.head(dec_mask_tokens)
        else:
            pred_mask_coord_values = self.aux_head(dec_mask_tokens)

        pred_mask_coord_values = pred_mask_coord_values.view(B,N,-1,3)

        if self.add_residual:
            pred_mask_coord_values += start

        if self.pred_all:
            pred_mask_coord_values = pred_mask_coord_values
            masked_gt = all_traj
        elif self.only_recons_past:
            pred_mask_coord_values = pred_mask_coord_values * (1 - mask_with_no_future[:,:,:,None])
            masked_gt = all_traj * (1 - mask_with_no_future[:,:,:,None])
        else:
            pred_mask_coord_values = pred_mask_coord_values * (1 - mask[:,:,:,None])
            masked_gt = all_traj * (1 - mask[:,:,:,None])

        if self.denoise_mode == 'all':
            dec_mask_tokens_noise = decoded_tokens_noise
        elif self.denoise_mode == 'past':
            dec_mask_tokens_noise = decoded_tokens_noise[batch_ind,agent_ind, past_ind, :]
        elif self.denoise_mode == 'future':
            dec_mask_tokens_noise = decoded_tokens_noise[batch_ind,agent_ind, future_ind, :]

        if self.same_head:
            if self.multi_output:
                pred_mask_coord_values_noise = self.head[-1](dec_mask_tokens_noise)
            else:
                pred_mask_coord_values_noise = self.head(dec_mask_tokens_noise)
        else:
            pred_mask_coord_values_noise = self.aux_head_2(dec_mask_tokens_noise)
        pred_mask_coord_values_noise = pred_mask_coord_values_noise.view(B,N,-1,3)

        if self.add_residual:
            pred_mask_coord_values_noise += start

        if self.only_recons_past:
            loss_mask = 1 - mask_with_no_future
        else:
            loss_mask = 1 - mask

        return pred_future_coord_values, pred_mask_coord_values,masked_gt, pred_mask_coord_values_noise, loss_mask

    def mask_forward(self,all_trajs,mask,all_out=False):
        B,N,T = all_trajs.shape[0],all_trajs.shape[1],all_trajs.shape[2]

        if self.concat_acc:
            vel = torch.zeros_like(all_trajs)
            vel[:,:,1:] = all_trajs[:,:,1:] - all_trajs[:,:,:-1]
            acc = torch.zeros_like(vel)
            acc[:,:,1:] = vel[:,:,1:] - vel[:,:,:-1]
            all_traj_input = torch.cat([all_trajs,vel,acc],dim=-1)
        elif self.concat_vel:
            vel = torch.zeros_like(all_trajs)
            vel[:,:,1:] = all_trajs[:,:,1:] - all_trajs[:,:,:-1]
            all_traj_input = torch.cat([all_trajs,vel],dim=-1)
        else:
            all_traj_input = all_trajs

        batch_ind = torch.arange(B)[:,None,None].cuda()
        agent_ind = torch.arange(N)[None,:,None].cuda()

        inverse_mask = 1 - mask
        unmask_tokens = self.patch_embed(all_traj_input) * mask[:,:,:,None]
        unmask_tokens += self.pos_embed.repeat(B,N,1,1)

        if self.add_joint_token:
            unmask_tokens += self.agent_embed.repeat(B,1,T,1)

        unmask_tokens = unmask_tokens * mask[:,:,:,None]

        mask_s = mask.permute(0,2,1).contiguous().view(B*T,N)
        mask_s = torch.matmul(mask_s[:,:,None],mask_s[:,None,:])
        mask_t = mask.contiguous().view(B*N,T)
        mask_t = torch.matmul(mask_t[:,:,None],mask_t[:,None,:])

        unmask_tokens_pad = unmask_tokens
        mask_ind = torch.arange(T)[None,None,:].repeat(B,N,1).cuda()

        out = []
        for l in range(self.encoder_depth):
            if l == 0:
                encoded_tokens = self.encoder[l](unmask_tokens_pad,mask_s,mask_t)

                enc_to_dec_tokens = encoded_tokens * mask[:,:,:,None]
                mask_tokens = self.mask_embed[None, None, None,:].repeat(B,N,T,1)
                mask_tokens += self.decoder_pos_embed(mask_ind)

                if self.add_joint_token:
                    mask_tokens += self.decoder_agent_embed(agent_ind.repeat(B,1,mask_tokens.shape[2]))
                
                mask_tokens = mask_tokens * inverse_mask[:,:,:,None]
                concat_tokens = enc_to_dec_tokens + mask_tokens

                dec_input_tokens = concat_tokens

                if self.decoder_masking:
                    decoded_tokens = self.decoder[l](dec_input_tokens,1-mask_s,1-mask_t)
                else:
                    decoded_tokens = self.decoder[l](dec_input_tokens)
                out.append(decoded_tokens)
            else:
                encoder_input = decoded_tokens

                encoder_input_pad = encoder_input * mask[:,:,:,None]
                encoder_output = self.encoder[l](encoder_input_pad,mask_s,mask_t)
                decoded_tokens = encoder_output * mask[:,:,:,None] + decoded_tokens * inverse_mask[:,:,:,None]
                if self.decoder_masking:
                    decoded_tokens = self.decoder[l](decoded_tokens,1-mask_s,1-mask_t)
                else:
                    decoded_tokens = self.decoder[l](decoded_tokens)
                out.append(decoded_tokens)
        if self.multi_output and all_out:
            return out
        else:
            return decoded_tokens

    def predict(self,all_traj):
        B,N,T = all_traj.shape[0],all_traj.shape[1],all_traj.shape[2]
        # assert N == 1
        all_traj = all_traj.view(B,N,T,3)
        start = all_traj[:,:,self.past_timestep-1:self.past_timestep]
        vel = torch.zeros_like(all_traj)
        vel[:,:,1:] = all_traj[:,:,1:] - all_traj[:,:,:-1]
        all_traj_with_vel = torch.cat([all_traj,vel],dim=-1)

        past_future_indices = torch.arange(T)[None].repeat(B,1).cuda()
        past_future_indices = past_future_indices[:,None,:].repeat(1,N,1)
        past_ind, future_ind = past_future_indices[:,:, :self.past_timestep], past_future_indices[:,:, self.past_timestep:]

        batch_ind = torch.arange(B)[:,None,None].cuda()
        agent_ind = torch.arange(N)[None,:,None].cuda()

        ordinary_mask = torch.zeros((B,N,T)).type_as(all_traj)
        ordinary_mask[:,:,:self.past_timestep] = 1.

        if self.multi_output:
            decoded_tokens = self.mask_forward(all_traj,ordinary_mask,all_out=True)[-1]
        else:
            decoded_tokens = self.mask_forward(all_traj,ordinary_mask)

        dec_future_tokens = decoded_tokens[batch_ind,agent_ind, future_ind, :]

        if self.multi_output:
            pred_future_coord_values = self.head[-1](dec_future_tokens)
        else:
            pred_future_coord_values = self.head(dec_future_tokens)

        pred_future_coord_values = pred_future_coord_values.view(B,N,-1,3)
        if self.add_residual:
            pred_future_coord_values += start
        return pred_future_coord_values

    def predict_with_mask(self,all_traj,unmask_ind):
        B,N,T = all_traj.shape[0],all_traj.shape[1],all_traj.shape[2]
        # assert N == 1
        all_traj = all_traj.view(B,N,T,3)
        start = all_traj[:,:,self.past_timestep-1:self.past_timestep]
        vel = torch.zeros_like(all_traj)
        vel[:,:,1:] = all_traj[:,:,1:] - all_traj[:,:,:-1]
        all_traj_with_vel = torch.cat([all_traj,vel],dim=-1)

        past_future_indices = torch.arange(T)[None].repeat(B,1).cuda()
        past_future_indices = past_future_indices[:,None,:].repeat(1,N,1)
        past_ind, future_ind = past_future_indices[:,:, :self.past_timestep], past_future_indices[:,:, self.past_timestep:]

        batch_ind = torch.arange(B)[:,None,None].cuda()
        agent_ind = torch.arange(N)[None,:,None].cuda()

        ordinary_mask = torch.zeros((B,N,T)).type_as(all_traj)
        # ordinary_mask[:,:,:self.past_timestep] = 1.

        ordinary_mask[batch_ind,agent_ind,unmask_ind] = 1.

        if self.multi_output:
            decoded_tokens = self.mask_forward(all_traj,ordinary_mask,all_out=True)[-1]
        else:
            decoded_tokens = self.mask_forward(all_traj,ordinary_mask)

        dec_future_tokens = decoded_tokens[batch_ind,agent_ind, future_ind, :]

        if self.multi_output:
            pred_future_coord_values = self.head[-1](dec_future_tokens)
            # pred_future_coord_values = self.head(dec_future_tokens)
        else:
            pred_future_coord_values = self.head(dec_future_tokens)

        pred_future_coord_values = pred_future_coord_values.view(B,N,-1,3)
        if self.add_residual:
            pred_future_coord_values += start
        return pred_future_coord_values

    def predict_with_noise_double(self,all_traj):
        B,N,T = all_traj.shape[0],all_traj.shape[1],all_traj.shape[2]
        # assert N == 1
        all_traj = all_traj.view(B,N,T,3)
        start = all_traj[:,:,self.past_timestep-1:self.past_timestep]
        # all_traj = all_traj - start
        vel = torch.zeros_like(all_traj)
        vel[:,:,1:] = all_traj[:,:,1:] - all_traj[:,:,:-1]
        all_traj_with_vel = torch.cat([all_traj,vel],dim=-1)

        past_future_indices = torch.arange(T)[None].repeat(B,1).cuda()
        past_future_indices = past_future_indices[:,None,:].repeat(1,N,1)
        past_ind, future_ind = past_future_indices[:,:, :self.past_timestep], past_future_indices[:,:, self.past_timestep:]

        batch_ind = torch.arange(B)[:,None,None].cuda()
        agent_ind = torch.arange(N)[None,:,None].cuda()

        ordinary_mask = torch.zeros((B,N,T)).type_as(all_traj)
        ordinary_mask[:,:,:self.past_timestep] = 1.

        if self.multi_output:
            decoded_tokens = self.mask_forward(all_traj,ordinary_mask,all_out=True)[-1]
        else:
            decoded_tokens = self.mask_forward(all_traj,ordinary_mask)

        dec_past_tokens = decoded_tokens[batch_ind,agent_ind, past_ind, :]

        pred_past_coord_values = self.aux_head_2(dec_past_tokens)

        pred_past_coord_values = pred_past_coord_values.view(B,N,-1,3)
        if self.add_residual:
            pred_past_coord_values += start

        ordinary_mask2 = torch.zeros((B,N,T)).type_as(all_traj)
        ordinary_mask2[:,:,:self.past_timestep] = 1.

        all_traj2 = torch.zeros_like(all_traj)
        all_traj2[:,:,:self.past_timestep] = pred_past_coord_values
        if self.multi_output:
            decoded_tokens = self.mask_forward(all_traj2,ordinary_mask2,all_out=True)[-1]
        else:
            decoded_tokens = self.mask_forward(all_traj2,ordinary_mask2)

        dec_future_tokens = decoded_tokens[batch_ind,agent_ind, future_ind, :]

        if self.multi_output:
            pred_future_coord_values = self.head[-1](dec_future_tokens)
            # pred_future_coord_values = self.head(dec_future_tokens)
        else:
            pred_future_coord_values = self.head(dec_future_tokens)

        pred_future_coord_values = pred_future_coord_values.view(B,N,-1,3)
        if self.add_residual:
            pred_future_coord_values += start

        return pred_future_coord_values


class TrajTrans_spatial(nn.Module):
    def __init__(
        self, num_patches, h_dim, depth=3, num_heads=8, mlp_dim=128,
        pool='cls', dim_per_head=64, dropout=0., embed_dropout=0.
    ):
        super().__init__()
        
        patch_dim = 3
        self.patch_embed = nn.Linear(patch_dim, h_dim)

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Add 1 for cls_token
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, h_dim)).cuda()

        self.dropout = nn.Dropout(p=embed_dropout)

        self.transformer_s = Transformer(
            h_dim, h_dim*2, depth=depth, num_heads=num_heads,
            dim_per_head=dim_per_head, dropout=dropout
        )

        self.transformer_t = Transformer(
            h_dim, h_dim*2, depth=depth, num_heads=num_heads,
            dim_per_head=dim_per_head, dropout=dropout
        )

        self.pool = pool

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
    
    def forward(self, patches):
        B, M, C = patches.shape

        '''ii. Patch embedding'''
        # (b,n_patches,dim)
        tokens = self.patch_embed(patches)
        # (b,n_patches+1,dim)
        tokens = torch.cat([self.cls_token.repeat(b, 1, 1), tokens], dim=1)
        tokens += self.pos_embed[:, :(num_patches + 1)]
        tokens = self.dropout(tokens)

        '''iii. Transformer Encoding'''
        enc_tokens = self.transformer(tokens)

        '''iv. Pooling'''
        # (b,dim)
        pooled = enc_tokens[:, 0] if self.pool == 'cls' else enc_tokens.mean(dim=1)

        '''v. Classification'''
        # (b,n_classes)
        logits = self.mlp_head(pooled)

        return logits

class STTrans(nn.Module):
    def __init__(
        self, num_patches, h_dim, depth=3, num_heads=8, mlp_dim=128,
        pool='cls', dim_per_head=64, dropout=0., embed_dropout=0., multi_output=False
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=embed_dropout)
        self.multi_output = multi_output

        self.transformer_s = []
        self.transformer_t = []
        self.depth = depth
        for i in range(depth):
            self.transformer_t.append(Transformer(
                h_dim, mlp_dim, depth=1, num_heads=num_heads,
                dim_per_head=dim_per_head, dropout=dropout
            ))
            self.transformer_s.append(Transformer(
                h_dim, mlp_dim, depth=1, num_heads=num_heads,
                dim_per_head=dim_per_head, dropout=dropout
            ))
        self.transformer_t = nn.ModuleList(self.transformer_t)
        self.transformer_s = nn.ModuleList(self.transformer_s)

        self.pool = pool

    def forward(self, x, mask_s=None, mask_t=None):
        B,N = x.shape[0], x.shape[1]
        out = []
        for i in range(self.depth):
            x = x.contiguous().view(B*N,-1,x.shape[-1])
            x = self.transformer_t[i](x,mask_t)
            x = x.view(B,N,-1,x.shape[-1]).permute(0,2,1,3)
            x = x.contiguous().view(-1,N,x.shape[-1])
            x = self.transformer_s[i](x,mask_s)
            x = x.view(B,-1,N,x.shape[-1]).permute(0,2,1,3)
            out.append(x)
        if self.multi_output:
            return out
        else:
            return x

class TrajTrans(nn.Module):
    def __init__(
        self, num_patches, h_dim, depth=3, num_heads=8, mlp_dim=128,
        pool='cls', dim_per_head=64, dropout=0., embed_dropout=0.
    ):
        super().__init__()
        
        patch_dim = 3
        self.patch_embed = nn.Linear(patch_dim, h_dim)

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Add 1 for cls_token
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, h_dim)).cuda()

        self.dropout = nn.Dropout(p=embed_dropout)

        self.transformer = Transformer(
            h_dim, h_dim*2, depth=depth, num_heads=num_heads,
            dim_per_head=dim_per_head, dropout=dropout
        )

        self.pool = pool

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
    
    def forward(self, patches):
        B, M, C = patches.shape

        '''ii. Patch embedding'''
        # (b,n_patches,dim)
        tokens = self.patch_embed(patches)
        # (b,n_patches+1,dim)
        tokens = torch.cat([self.cls_token.repeat(b, 1, 1), tokens], dim=1)
        tokens += self.pos_embed[:, :(num_patches + 1)]
        tokens = self.dropout(tokens)

        '''iii. Transformer Encoding'''
        enc_tokens = self.transformer(tokens)

        '''iv. Pooling'''
        # (b,dim)
        pooled = enc_tokens[:, 0] if self.pool == 'cls' else enc_tokens.mean(dim=1)

        '''v. Classification'''
        # (b,n_classes)
        logits = self.mlp_head(pooled)

        return logits

def to_pair(t):
    return t if isinstance(t, tuple) else (t, t)

 
class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = net
    
    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x, mask=None):
        b, l, d = x.shape

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv = self.to_qkv(x)
        # (3,b,num_heads,l,dim_per_head)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        q, k, v = qkv.chunk(3)

        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        '''ii. Attention computation'''
        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        if mask is not None:
            mask = mask[:,None,:,:].repeat(1,self.num_heads,1,1)
            attn = attn * mask
            attn = attn / (torch.sum(attn,dim=-1,keepdim=True)+1e-10)

        '''iii. Put attention on Value & reshape'''
        # (b,num_heads,l,dim_per_head)
        z = torch.matmul(attn, v)

        # (b,num_heads,l,dim_per_head)->(b,l,num_heads,dim_per_head)->(b,l,dim_all_heads)
        z = z.transpose(1, 2).reshape(b, l, -1)
        # assert z.size(-1) == q.size(-1) * self.num_heads
        '''iv. Project out'''
        # (b,l,dim_all_heads)->(b,l,dim)
        out = self.out(z)
        # assert out.size(-1) == d
        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth=1, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x, mask=None):
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x,mask=mask)
            x = x + norm_ffn(x)
        
        return x

class Transformer_joint(nn.Module):
    def __init__(self, dim, mlp_dim, depth=2, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask_s=None, mask_t=None, mask_st=None):
        for norm_attn_s,norm_attn_t,norm_attn_st, norm_ffn in self.layers:
            x = x + norm_attn_s(x,mask=mask_s) + norm_attn_t(x,mask=mask_t) + norm_attn_st(x,mask=mask_st)
            x = x + norm_ffn(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

