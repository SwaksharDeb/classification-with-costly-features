import numpy as np
from consts import *

from ipdb import set_trace

LIN_ARRAY = np.arange(AGENTS)

class Environment:
	def __init__(self, data, label, init, ff):
		self.data_x = data
		self.data_y = label
		self.data_init = init
		self.data_len = len(data)

		self.cost = np.ones(AGENTS)

		self.mask = np.zeros( (AGENTS, FEATURE_DIM+CONTEXT_DIM), dtype=np.float32 )
		self.x    = np.zeros( (AGENTS, FEATURE_DIM+CONTEXT_DIM), dtype=np.float32 )
		self.y    = np.zeros( AGENTS, dtype=int )

		self.ff = ff

	def reset(self):
		for i in range(AGENTS):
			self._reset(i)

		return self._get_state()

	def _reset(self, i):
		self.mask[i] = 0
		self.mask[i][-10:] = 1
		self.x[i], self.y[i], init = self._generate_sample()
		self.mask[i][init[0]] = 1

	def step(self, action):
		queryed = action - CLASSES
		queryed = queryed.clip(min=-1)
		s = self._get_state()

		self.mask[LIN_ARRAY, queryed] = 1
		query_num = self.mask.sum(1)-11
		finish_mask = (query_num==QUERY_BUDGET)

		r = -self.cost * self.ff

		for i in np.where(finish_mask==True)[0]:
			r[i] = REWARD_INCORRECT
			self._reset(i)

		for i in np.where(action < CLASSES)[0]:
			r[i] = REWARD_CORRECT if action[i] == self.y[i] else REWARD_INCORRECT
			self._reset(i)

		s_ = self._get_state()
		num_s_one  = (s ==1).sum(1)-3
		num_s__one = (s_==1).sum(1)-3
		addition_reward = SHAPING_FACTOR * (1 * num_s__one - num_s_one)
		r = r+addition_reward

		return (s_, r)

	def _generate_sample(self):
		idx = np.random.randint(0, self.data_len)
		x = self.data_x[idx]
		y = self.data_y[idx]
		init = self.data_init[idx]
		return (x, y, init)

	def _get_state(self):
		x_ = self.x * self.mask
		return x_
		