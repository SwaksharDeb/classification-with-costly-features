#%% variables are initialized in const.py file
"""
    AGENT = how many samples to collect in one step= 1000
    EPSILON_START = 0.50
    CLASSES = 2 = as we are loading miniboones dataset(mb) by default, it has 2 classes(0 and 1)
    FEATURE_DIM = 50 = as we are loading miniboones dataset(mb) by default, it has 50 features
    ACTION_DIM = FEATURE_DIM + CLASSES
    STATE_DIM = FEATURE_DIM * 2( multiply by 2 because of the mask)
              = First two coloum of the SATE_DIM specify classes(assuming miniboone dataset is loaded by deffault)
"""
#%%

import numpy as np
import torch

from consts import *

class Pool():
    def __init__(self, size):
        """
        initialize states, actions, rewards, next states tensors
        Params:
            size = POOL_SIZE = 2000000
                = size of the memory buffer
        """
        self.data_s  = torch.FloatTensor(size, STATE_DIM)
        self.data_a  = torch.LongTensor(size)
        self.data_r  = torch.FloatTensor(size)
        self.data_s_ = torch.FloatTensor(size, STATE_DIM)

        self.idx  = 0
        self.size = size

    def put(self, x):
        """
        Store state, next state, actions and rewards tensor into replay memory
        Params:
            x = sampled experiences = states, actions, rewards, next states
        """
        s, a, r, s_ = x
        size = len(s)

        self.data_s [self.idx:self.idx+size] = torch.from_numpy(s)
        self.data_a [self.idx:self.idx+size] = torch.from_numpy(a)
        self.data_r [self.idx:self.idx+size] = torch.from_numpy(r)
        self.data_s_[self.idx:self.idx+size] = torch.from_numpy(s_)

        self.idx = (self.idx + size) % self.size  # circular queue

    def sample(self, size):
        """
        sample ranodm index number from memory
        """
        idx = torch.from_numpy(np.random.choice(self.size, size)).type(torch.long).cuda()
        return self.data_s[idx], self.data_a[idx], self.data_r[idx], self.data_s_[idx]

    def cuda(self):
        """
        store tensors states, actions, rewards and next states into GPU
        """
        self.data_s  = self.data_s.cuda()
        self.data_a  = self.data_a.cuda()
        self.data_r  = self.data_r.cuda()
        self.data_s_ = self.data_s_.cuda()