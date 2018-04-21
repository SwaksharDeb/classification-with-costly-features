import visdom
import numpy as np
import time, sys, utils

from consts import *


#==============================
class PerfAgent():
	def __init__(self, env, brain):
		self.env  = env
		self.brain = brain

		self.agents = self.env.agents

		self.done = np.zeros(self.agents, dtype=np.bool)
		self.total_r = np.zeros(self.agents)
		self.total_len = np.zeros(self.agents, dtype=np.int32)
		self.total_corr  = np.zeros(self.agents, dtype=np.int32)

		self.s = self.env.reset()

	def act(self, s):
		m = np.zeros((self.agents, ACTION_DIM))	# create max_mask
		m[:, CLASSES:] = (s[:, :FEATURE_DIM]!=0).astype(np.float32)

		p = self.brain.predict_np(s) - MAX_MASK_CONST * m 	# select an action not considering those already performed
		a = np.argmax(p, axis=1)

		return a

	def step(self):
		a = self.act(self.s)
		s_, r, done = self.env.step(a)
		self.s = s_

		newly_finished = ~self.done & done
		self.done = self.done | done

		self.total_r   = self.total_r   + r * (newly_finished | ~done)
		self.total_len = self.total_len + ~done
		self.total_corr = self.total_corr + (r == REWARD_CORRECT) * newly_finished

	def run(self):
		while not np.all(self.done):
			self.step()
		avg_r    = np.mean(self.total_r)
		avg_len  = np.mean(self.total_len)
		avg_corr = np.mean(self.total_corr)
		return avg_r, avg_len, avg_corr

#==============================
class PerfEnv:
	def __init__(self, data, label, init, ff):
		self.x = data.numpy()
		self.y = label.numpy()
		self.init = init.numpy()

		self.agents = self.x.shape[0]
		self.lin_array = np.arange(self.agents)
		self.cost = np.ones(self.agents)
		self.ff = ff

	def reset(self):
		self.mask = np.zeros( (self.agents, STATE_DIM), dtype=np.float32 )
		for i in range(self.agents):
			self.mask[i][-10:] = 1
			self.mask[i][self.init[i]] = 1

		self.done = np.zeros( self.agents, dtype=np.bool )

		return self._get_state()

	def step(self, action):
		queryed = action - CLASSES
		queryed = queryed.clip(min=-1)
		s = self._get_state()

		self.mask[self.lin_array, queryed] = 1
		query_num = self.mask.sum(1)-11
		finish_mask = (query_num==QUERY_BUDGET)

		r = -self.ff * self.cost

		for i in np.where(finish_mask==True)[0]:
			r[i] = REWARD_INCORRECT
			self.done[i] = 1

		for i in np.where(action < CLASSES)[0]:
			r[i] = REWARD_CORRECT if action[i] == self.y[i] else REWARD_INCORRECT
			self.done[i] = 1

		s_ = self._get_state()
		num_s_one  = (s ==1).sum(1)-3
		num_s__one = (s_==1).sum(1)-3
		addition_reward = SHAPING_FACTOR * (1 * num_s__one - num_s_one)
		r = r+addition_reward

		return (s_, r, self.done)

	def _get_state(self):
		x_ = self.x * self.mask
		return x_
		
#==============================
class Log:
	def __init__(self, data_val, label_val, init_val, ff, brain):
		self.env = PerfEnv(data_val, label_val, init_val, ff)
		self.brain = brain

		if PLOT:
			self.vis = visdom.Visdom()
			title_accuracy = 'Accuracy ({} dis rshape {})'.format(CLASSES, SHAPING_FACTOR)
			title_reward   = 'Reward ({} dis rshape {})'.format(CLASSES, SHAPING_FACTOR)
			title_steps    = 'Steps ({} dis rshape {})'.format(CLASSES, SHAPING_FACTOR)

			self.avg_accuracy_win = self.vis.line(np.array([np.nan]), opts=dict(title=title_accuracy), env=PLOT_ENV)
			self.avg_reward_win   = self.vis.line(np.array([np.nan]), opts=dict(title=title_reward), env=PLOT_ENV)
			self.avg_steps_win    = self.vis.line(np.array([np.nan]), opts=dict(title=title_steps), env=PLOT_ENV)
			
		if BLANK_INIT:
			mode = "w"
		else:
			mode = "a"

		self.perf_file = open("run_perf.dat", mode)
		self.time = 0


	def log_perf(self, epoch):
		agent = PerfAgent(self.env, self.brain)
		avg_r, avg_step, avg_corr = agent.run()

		print("{:.3f} {:.3f} {:.3f}".format(avg_r, avg_step, avg_corr), file=self.perf_file, flush=True)
		print("[epoch {}] reward:{:.3f}\tasked:{:.3f}\taccuracy:{:.3f}".format(epoch, avg_r, avg_step, avg_corr))
		if PLOT:
			self.plot(epoch, avg_r, avg_step, avg_corr)

	def plot(self, epoch, avg_r, avg_step, avg_corr):
		self.vis.line(
            np.array([avg_corr]),
            np.array([epoch]),
            env=PLOT_ENV,
            win=self.avg_accuracy_win,
            update='append'
        )
		self.vis.line(
            np.array([avg_step]),
            np.array([epoch]),
            env=PLOT_ENV,
            win=self.avg_steps_win,
            update='append'
        )
		self.vis.line(
            np.array([avg_r]),
            np.array([epoch]),
            env=PLOT_ENV,
            win=self.avg_reward_win,
            update='append'
        )
		self.vis.save([PLOT_ENV])
