from consts import *

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ipdb import set_trace
#==============================
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        in_nn  = STATE_DIM
        out_nn = NN_FC_DENSITY

        self.l_fc = []
        for i in range(NN_HIDDEN_LAYERS):
            l = torch.nn.Linear(in_nn, out_nn)
            in_nn = out_nn

            self.l_fc.append(l)
            self.add_module("l_fc_"+str(i), l)

        self.l_out_q_val = torch.nn.Linear(in_nn, ACTION_DIM)        # q-value prediction

        self.opt = torch.optim.RMSprop(self.parameters(), lr=OPT_LR, alpha=OPT_ALPHA)

        self.loss_f = torch.nn.MSELoss()

        self.cuda()

    def forward(self, batch):
        """
        Params:
            batch = batch of input states = FEATURE_DIM*2
        """
        flow = batch

        for l in self.l_fc:
            flow = F.relu(l(flow))

        a_out_q_val = self.l_out_q_val(flow)

        return a_out_q_val

    def copy_weights(self, other, rho=TARGET_RHO):
        """
        Soft update
        """
        params_other = list(other.parameters())
        params_self  = list(self.parameters())

        for i in range( len(params_other) ):
            val_self  = params_self[i].data
            val_other = params_other[i].data
            val_new   = rho * val_other + (1-rho) * val_self

            params_self[i].data.copy_(val_new)

    def train_network(self, s, action, q_):
        """
        We are considering gamma = 1
        Params:
            s = batch of input states = FEATURE_DIM *2
            action = batch of actions performed
            q_ = batch of TD errors
        Outputs:
            adjust the network parameters to minimize loss
        """
        s  = Variable(s)
        action  = Variable(action)
        q_ = Variable(q_)
        ## self(s) == call forward function
        q_pred = self(s).gather(1, action.view(-1,1))    # we have results only for performed actions
        loss_q = self.loss_f(q_pred, q_)

        self.opt.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.parameters(), OPT_MAX_NORM)
        loss_q.backward()
        self.opt.step()

    def set_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr