# %% variables are initialized in const.py file
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
from torch.autograd import Variable
import sys
from consts import *
from net import Net

#==============================
class Brain:
    def __init__(self, pool):
        self.pool = pool

        self.model  = Net() # local network
        self.model_ = Net() # target network

        print("Network architecture:\n"+str(self.model))

    def _load(self):
        self.model.load_state_dict( torch.load("model") )  # loading the local network
        self.model_.load_state_dict( torch.load("model_") ) # loading the target network

    def _save(self):
        torch.save(self.model.state_dict(), "model") # saving the local network
        torch.save(self.model_.state_dict(), "model_") # saving the target network

    def predict_pt(self, s, target):
        """
        Params:
            s = input state = FEATURE_DIM * 2
        Outputs:
            predicted Q valus of the state(pt stands for pytorch tensor prediction)
        """
        s = Variable(s)

        if target:
            return self.model_(s).data # model_ = target network
        else:
            return self.model(s).data # model = locak network

    def predict_np(self, s, target=False):
        """
        Params:
            s = input state = FEATURE_DIM*2
        Outputs:
            predicted Q vaues of the state and convert to numpy array(np stands for numpy prediction)
        """
        s = torch.from_numpy(s).cuda()
        res = self.predict_pt(s, target)
        return res.cpu().numpy()

    def train(self):
        """
            We use double Q learning algorithm
        """
        s, a, r, s_ = self.pool.sample(BATCH_SIZE) # returens (states, actions, rewards, net_states)

        # extract the mask
        m_ = torch.FloatTensor(BATCH_SIZE, ACTION_DIM).zero_().cuda() # variable m stands for extracting self.mask Variable
        m_[:, CLASSES:] = s_[:, FEATURE_DIM:]

        # compute
        q_current = self.predict_pt(s_, target=False) - (MAX_MASK_CONST * m_) # masked actions do not influence the max
        q_target  = self.predict_pt(s_, target=True)

        # Double Q learning
        _, amax = q_current.max(1, keepdim=True) #selecting the greedy action w.r.t local network
        q_ = q_target.gather(1, amax) #evaluating the greedy action w.r.t target network

        # when a<CLASSES: predict label (q is terminal step) q_ = 0
        q_[ a < CLASSES ] = 0
        q_ = q_ + r.view(-1,1)  #discount factor = 1, q_ is the TD target

        # bound the values to theoretical q function range
        q_.clamp_(-1, 0)

        self.model.train_network(s, a, q_)
        self.model_.copy_weights(self.model)  # soft update

    def update_lr(self, epoch):
        """
        Learning rate scheduler
        """
        lr = OPT_LR * (LR_SC_FACTOR ** (epoch // LR_SC_EPOCHS))
        lr = max(lr, LR_SC_MIN)

        self.model.set_lr(lr)
        print("Setting LR:", lr)
