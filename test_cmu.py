import argparse
import torch
from dataset.dataloader import CMU_Motion3D
from model.model import AuxFormer
from torch import nn, optim
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import yaml
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def lr_decay(optimizer, lr_now, gamma):
    lr_new = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new

def main():
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0,1000)
        setup_seed(seed)
    print('The seed is :',seed)

    past_length = args.past_length
    future_length = args.future_length

    if args.debug:
        dataset_train = CMU_Motion3D(actions='walking', input_n=args.past_length, output_n=args.future_length, split=0, scale=args.scale,downsample=args.downsample)
    else:
        dataset_train = CMU_Motion3D(actions='all', input_n=args.past_length, output_n=args.future_length, split=0, scale=args.scale,downsample=args.downsample)
        
    acts = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
               "washwindow"]
    loaders_test = {}
    for act in acts:
        dataset_test = CMU_Motion3D(actions=act, input_n=args.past_length, output_n=args.future_length, split=1, scale=args.scale,dim_used = dataset_train.dim_used,downsample=args.downsample)
        loaders_test[act] = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=8)

    dim_used = dataset_test.dim_used

    model = AuxFormer(in_dim=2, 
                    h_dim=args.nf, 
                    past_timestep=args.past_length,
                    future_timestep=args.future_length, 
                    mask_ratio=args.mask_ratio, 
                    decoder_dim=args.decoder_dim, 
                    num_heads=8, 
                    encoder_depth=args.encoder_depth,
                    decoder_depth=args.decoder_depth,
                    decoder_dim_per_head=args.dim_per_head,
                    same_head=args.same_head,
                    range_mask_ratio=args.range_mask_ratio,
                    mlp_head=args.mlp_head,
                    mask_past=args.mask_past,
                    mask_range=args.mask_range,
                    multi_output=args.multi_output,
                    decoder_masking=args.decoder_masking,
                    pred_all=args.pred_all,
                    mlp_dim=args.mlp_dim,
                    dim_per_head=args.dim_per_head,
                    noise_dev=args.noise_dev,
                    part_noise=args.part_noise,
                    denoise_mode=args.denoise_mode,
                    part_noise_ratio=args.part_noise_ratio,
                    add_joint_token=args.add_joint_token,
                    n_agent=25,
                    concat_vel=args.concat_vel,
                    concat_acc=args.concat_acc,
                    only_recons_past=args.only_recons_past,
                    add_residual=args.add_residual,
                    denoise=args.denoise,
                    regular_masking=args.regular_masking,
                    multi_same_head=args.multi_same_head,
                    range_noise_dev=args.range_noise_dev
    )

    model = model.cuda()

    # def get_parameter_number(model):
    #     total_num = sum(p.numel() for p in model.parameters())
    #     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return {'Total': total_num, 'Trainable': trainable_num}
    # print(get_parameter_number(model))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_path = args.model_save_dir + '/' + args.model_save_name +'.pth.tar'
    print('Loading model from:', model_path)
    model_ckpt = torch.load(model_path)
    model.load_state_dict(model_ckpt['state_dict'])
    model.eval()

    if args.future_length == 25 or args.future_length == 13:
        avg_mpjpe = np.zeros((6))
    elif args.future_length == 15:
        avg_mpjpe = np.zeros((2))
    else:
        avg_mpjpe = np.zeros((4))
    for act in acts:
        mpjpe = test(model, optimizer, 0, (act, loaders_test[act]), dim_used, backprop=False)
        avg_mpjpe += mpjpe
    avg_mpjpe = avg_mpjpe / len(acts)
    print('avg mpjpe:',avg_mpjpe)

    return

def train(model, optimizer, epoch, loader, dim_used=[], backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, length, _ = data[0].size()
        data = [d.to(device) for d in data]
        loc, vel, loc_end, _, item = data
        loc_start = loc[:,:,-1:]

        optimizer.zero_grad()

        if args.time_exp:
            torch.cuda.synchronize()
            t1 = time.time()
        all_traj = torch.cat([loc,loc_end],dim=2)

        loc_pred,mask_pred,mask_gt,denoised_pred, mask = model(all_traj)

        if args.weighted_loss:
            weight = np.arange(1,args.max_weight,((args.max_weight-1)/args.future_length))
            weight = args.future_length / weight
            weight = torch.from_numpy(weight).type_as(loc_end)
            weight = weight[None,None]
        else:
            weight = 1

        if args.multi_output:
            loss = 0
            for idx,item in enumerate(loc_pred):
                loss += (torch.mean(weight*torch.norm(item-loc_end,dim=-1))/args.encoder_depth)
        else:
            loss = torch.mean(weight*torch.norm(loc_pred-loc_end,dim=-1))

        loss += torch.sum(torch.norm(mask_pred-mask_gt,dim=-1,p=2))/torch.sum(mask)
        if args.denoise_mode == 'all':
            loss += torch.mean(torch.norm(denoised_pred-all_traj,dim=-1))
        elif args.denoise_mode == 'past':
            loss += torch.mean(torch.norm(denoised_pred-loc,dim=-1))
        elif args.denoise_mode == 'future':
            loss += torch.mean(torch.norm(denoised_pred-loc_end,dim=-1))
        else:
            raise ValueError("args.denoise_mode")

        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size

    print('%s epoch %d avg loss: %.5f' % ('train', epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']

def test(model, optimizer, epoch, act_loader,dim_used=[],backprop=False):
    act, loader = act_loader[0], act_loader[1]

    model.eval()

    validate_reasoning = False
    if validate_reasoning:
        acc_list = [0]*args.n_layers
    res = {'epoch': epoch, 'loss': 0, 'counter': 0}

    output_n = args.future_length
    if output_n == 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 15:
        eval_frame = [3, 14]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    elif output_n == 13:
        eval_frame = [0,1,3,4,6,12]
    t_3d = np.zeros(len(eval_frame))

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            batch_size, n_nodes, length, _ = data[0].size()
            data = [d.to(device) for d in data]
            loc, vel, loc_end, loc_end_ori, _ = data
            loc_start = loc[:,:,-1:]
            pred_length = loc_end.shape[2]

            optimizer.zero_grad()

            loc_end_fake = torch.zeros_like(loc_end)
            all_traj = torch.cat([loc,loc_end_fake],dim=2)
            loc_pred = model.predict(all_traj) #(B,N,T,3)

            pred_3d = loc_end_ori.clone()
            loc_pred = loc_pred.transpose(1,2)
            loc_pred = loc_pred.contiguous().view(batch_size,loc_end.shape[2],n_nodes*3)
            
            joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
            index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
            index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

            pred_3d[:,:,dim_used] = loc_pred
            pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
            pred_p3d = pred_3d.contiguous().view(batch_size, pred_length, -1, 3) #[:, input_n:, :, :]
            targ_p3d = loc_end_ori.contiguous().view(batch_size, pred_length, -1, 3) #[:, input_n:, :, :]

            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                t_3d[k] += torch.mean(torch.norm(targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2, 1)).item() * batch_size
            
            res['counter'] += batch_size

    t_3d *= args.scale
    N = res['counter']
    actname = "{0: <14} |".format(act)
    if args.future_length == 25 or args.future_length == 13:
        print('Act: {},  ErrT: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}, TestError {:.4f}'\
            .format(actname, 
                    float(t_3d[0])/N, float(t_3d[1])/N, float(t_3d[2])/N, float(t_3d[3])/N, float(t_3d[4])/N, float(t_3d[5])/N, 
                    float(t_3d.mean())/N))
    elif args.future_length == 15:
        print('Act: {},  ErrT: {:.3f} {:.3f}, TestError {:.4f}'\
            .format(actname, 
                    float(t_3d[0])/N, float(t_3d[1])/N, 
                    float(t_3d.mean())/N))
    else:
        print('Act: {},  ErrT: {:.3f} {:.3f} {:.3f} {:.3f}, TestError {:.4f}'\
            .format(actname, 
                    float(t_3d[0])/N, float(t_3d[1])/N, float(t_3d[2])/N, float(t_3d[3])/N, 
                    float(t_3d.mean())/N))

    return t_3d / N

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=-1, metavar='S', help='random seed (default: -1)')
    parser.add_argument("--debug",action='store_true')
    parser.add_argument('--model_save_dir', type=str, default='ckpt', help='dir to save model')
    parser.add_argument("--model_save_name",type=str,default="default")
    parser.add_argument("--task",type=str,default="short")
    args = parser.parse_args()

    if args.task == 'short':
        with open('cfg/cmu_short.yml', 'r') as f:
            yml_arg = yaml.load(f)
    else:
        with open('cfg/cmu_long.yml', 'r') as f:
            yml_arg = yaml.load(f)

    parser.set_defaults(**yml_arg)
    args = parser.parse_args()
    args.cuda = True

    device = torch.device("cuda" if args.cuda else "cpu")

    print(args)
    main()




