from consts import *

import torch
import torch.nn as nn
from torch.autograd import Variable


#==============================
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.layer_size = [STATE_DIM] + HIDDEN_DIM + [ACTION_DIM]

		self.linears = nn.ModuleList(
			[nn.Linear(self.layer_size[i], self.layer_size[i + 1])
				for i in range(len(self.layer_size) - 1)])
		self.activations = nn.ModuleList(
			[nn.ReLU() for i in range(len(self.layer_size) - 2)])

		self.opt = torch.optim.RMSprop(self.parameters(), lr=OPT_LR, alpha=OPT_ALPHA)

		self.loss_f = torch.nn.MSELoss()

		self.cuda()

	def forward(self, batch):
		flow = batch
		for i in range(len(self.layer_size) - 2):
			flow = self.linears[i](flow)
			flow = self.activations[i](flow)

		a_out_q_val = self.linears[-1](flow)

		return a_out_q_val

	def copy_weights(self, other, rho=TARGET_RHO):
		params_other = list(other.parameters())
		params_self  = list(self.parameters())

		for i in range( len(params_other) ):
			val_self  = params_self[i].data
			val_other = params_other[i].data
			val_new   = rho * val_other + (1-rho) * val_self

			params_self[i].data.copy_(val_new)

	def train_network(self, s, a, q_):
		s  = Variable(s)
		a  = Variable(a)
		q_ = Variable(q_)

		## self(s) == call forward function
		q_pred = self(s).gather(1, a)	# we have results only for performed actions
		loss_q = self.loss_f(q_pred, q_)

		self.opt.zero_grad()
		torch.nn.utils.clip_grad_norm(self.parameters(), OPT_MAX_NORM)
		loss_q.backward()
		self.opt.step()

	def set_lr(self, lr):
		for param_group in self.opt.param_groups:
			param_group['lr'] = lr