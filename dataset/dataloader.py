from torch.utils.data import Dataset
import numpy as np
from dataset import data_utils
import random
import os
import pickle as pkl
from os import walk

class H36motion3D(Dataset):
    def __init__(self, actions='all', input_n=10, output_n=10, split=0, scale=100, sample_rate=2):
        """
        param split: 0 train, 1 testing, 2 validation
        """
        path_to_data = 'data/h3.6m/dataset'
        self.path_to_data = path_to_data
        self.split = split
        self.input_n = input_n
        self.output_n = output_n

        if output_n == 13:
            output_n = 25
            self.output_n = 25
            self.downsample = True
        else:
            self.downsample = False
            
        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])

        acts = data_utils.define_actions(actions)
            
        subjs = subs[split]
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate, input_n + output_n)
        all_seqs = all_seqs / scale

        self.all_seqs_ori = all_seqs.copy()
        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used] # （B,T,N*3）
        all_seqs = all_seqs.reshape(all_seqs.shape[0],all_seqs.shape[1],-1,3) #(B,T,N,3)
        all_seqs = all_seqs.transpose(0,2,1,3) #(B,N,T,3)

        all_seqs_vel = np.zeros_like(all_seqs)
        all_seqs_vel[:,:,1:] = all_seqs[:,:,1:] - all_seqs[:,:,:-1]
        all_seqs_vel[:,:,0] = all_seqs_vel[:,:,1]

        self.all_seqs = all_seqs
        self.all_seqs_vel = all_seqs_vel


    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        loc_data = self.all_seqs[item]
        vel_data = self.all_seqs_vel[item]
        loc_data_ori = self.all_seqs_ori[item] #(T,N3)
        if self.downsample:
            output = loc_data[:,1+self.input_n:self.input_n+self.output_n:2]
            output = np.concatenate([output,loc_data[:,-1:]],axis=1)
            output_ori = loc_data_ori[1+self.input_n:self.input_n+self.output_n:2]
            output_ori = np.concatenate([output_ori,loc_data_ori[-1:]],axis=0)
            return loc_data[:,:self.input_n], vel_data[:,:self.input_n],output,output_ori,item
        else:
            return loc_data[:,:self.input_n], vel_data[:,:self.input_n], loc_data[:,self.input_n:self.input_n+self.output_n],loc_data_ori[self.input_n:self.input_n+self.output_n],item

class CMU_Motion3D(Dataset):
    def __init__(self, actions, input_n=10, output_n=10, split=0, dim_used=0, scale=100,downsample=False):
        path_to_data = 'data/CMUMocap'
        self.path_to_data = path_to_data
        self.split = split
        self.input_n = input_n
        if output_n == 15:
            self.skip = True
            output_n = 25
            self.output_n = 25
            self.downsample = False
        elif output_n == 13:
            self.downsample = True
            output_n = 25
            self.output_n = 25
            self.skip = False
        else:
            self.skip = False
            self.downsample = False
            self.output_n = output_n
        actions = data_utils.define_actions_cmu(actions)

        if split == 0:
            path_to_data = os.path.join(path_to_data, 'train')
            is_test = False
        else:
            path_to_data = os.path.join(path_to_data, 'test')
            is_test = True

        all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d(path_to_data, actions,
                                                                    input_n, output_n,
                                                                    is_test=is_test)
        if not is_test:
            dim_used = dim_use

        all_seqs = all_seqs / scale
        self.all_seqs_ori = all_seqs.copy()

        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used]

        all_seqs = all_seqs.reshape(all_seqs.shape[0],all_seqs.shape[1],-1,3) #(B,T,N,3)
        all_seqs = all_seqs.transpose(0,2,1,3) #(B,N,T,3)

        all_seqs_vel = np.zeros_like(all_seqs)
        all_seqs_vel[:,:,1:] = all_seqs[:,:,1:] - all_seqs[:,:,:-1]
        all_seqs_vel[:,:,0] = all_seqs_vel[:,:,1]

        self.all_seqs = all_seqs
        self.all_seqs_vel = all_seqs_vel

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        loc_data = self.all_seqs[item]
        vel_data = self.all_seqs_vel[item]
        loc_data_ori = self.all_seqs_ori[item] #(T,N3)
        if self.downsample:
            output = loc_data[:,1+self.input_n:self.input_n+self.output_n:2]
            output = np.concatenate([output,loc_data[:,-1:]],axis=1)
            output_ori = loc_data_ori[1+self.input_n:self.input_n+self.output_n:2]
            output_ori = np.concatenate([output_ori,loc_data_ori[-1:]],axis=0)
            return loc_data[:,:self.input_n], vel_data[:,:self.input_n],output,output_ori,item
        elif self.skip:
            return loc_data[:,:self.input_n], vel_data[:,:self.input_n], loc_data[:,self.input_n+10:self.input_n+self.output_n],loc_data_ori[self.input_n+10:self.input_n+self.output_n],item
        else:
            return loc_data[:,:self.input_n], vel_data[:,:self.input_n], loc_data[:,self.input_n:self.input_n+self.output_n],loc_data_ori[self.input_n:self.input_n+self.output_n],item

class Pose3dPW3D(Dataset):
    def __init__(self, input_n=10, output_n=30, dct_n=15, split=0, scale=100, debug=False):
        path_to_data = 'data/3DPW/sequenceFiles'
        self.path_to_data = path_to_data
        self.split = split
        self.dct_n = dct_n
        self.input_n = input_n
        if output_n == 10:
            output_n = 30
            self.output_n = 30
        else:
            self.output_n = output_n
        # since baselines (http://arxiv.org/abs/1805.00655.pdf and https://arxiv.org/pdf/1705.02445.pdf)
        # use observed 50 frames but our method use 10 past frames in order to make sure all methods are evaluated
        # on same sequences, we first crop the sequence with 50 past frames and then use the last 10 frame as input
        if split == 1:
            their_input_n = 50
        else:
            their_input_n = input_n
        seq_len = their_input_n + output_n

        if split == 0:
            self.data_path = path_to_data + '/train/'
        elif split == 1:
            self.data_path = path_to_data + '/test/'
        elif split == 2:
            self.data_path = path_to_data + '/validation/'
        all_seqs = []
        files = []
        for (dirpath, dirnames, filenames) in walk(self.data_path):
            files.extend(filenames)
        if debug:
            used = 1
        else:
            used = len(files)
        for f in files[:used]:
            with open(self.data_path + f, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['jointPositions']
                for i in range(len(joint_pos)):
                    seqs = joint_pos[i]
                    seqs = seqs - seqs[:, 0:3].repeat(24, axis=0).reshape(-1, 72)
                    n_frames = seqs.shape[0]
                    fs = np.arange(0, n_frames - seq_len + 1)
                    fs_sel = fs
                    for j in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + j + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = seqs[fs_sel, :]
                    if len(all_seqs) == 0:
                        all_seqs = seq_sel
                    else:
                        all_seqs = np.concatenate((all_seqs, seq_sel), axis=0)

        all_seqs = all_seqs * 1000
        all_seqs = all_seqs[:, (their_input_n - input_n):, :]

        print(all_seqs.shape)

        all_seqs = all_seqs / scale
        self.all_seqs_ori = all_seqs.copy()

        self.dim_used = np.array(range(3, all_seqs.shape[2]))
        all_seqs = all_seqs[:, :, 3:]
        n, seq_len, dim_len = all_seqs.shape

        all_seqs = all_seqs.reshape(all_seqs.shape[0],all_seqs.shape[1],-1,3) #(B,T,N,3)
        all_seqs = all_seqs.transpose(0,2,1,3) #(B,N,T,3)
        # print(all_seqs.shape)
        all_seqs_vel = np.zeros_like(all_seqs)
        all_seqs_vel[:,:,1:] = all_seqs[:,:,1:] - all_seqs[:,:,:-1]
        all_seqs_vel[:,:,0] = all_seqs_vel[:,:,1]

        self.all_seqs = all_seqs
        self.all_seqs_vel = all_seqs_vel


    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        loc_data = self.all_seqs[item]
        vel_data = self.all_seqs_vel[item]
        loc_data_ori = self.all_seqs_ori[item] #(T,N3)

        if self.output_n == 30:
            return loc_data[:,:self.input_n], vel_data[:,:self.input_n], loc_data[:,2+self.input_n:self.input_n+self.output_n:3],loc_data_ori[2+self.input_n:self.input_n+self.output_n:3],item
        else:
            return loc_data[:,:self.input_n], vel_data[:,:self.input_n], loc_data[:,self.input_n:self.input_n+self.output_n],loc_data_ori[self.input_n:self.input_n+self.output_n],item
