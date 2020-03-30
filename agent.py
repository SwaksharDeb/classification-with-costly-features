#%% variables are initialized in const.py file
"""
    AGENT = how many samples to collect in one step= 1000
    EPSILON_START = 0.50
    CLASSES = 2 = as we are loading miniboones dataset(mb) by default, it has 2 classes(0 and 1)
    FEATURE_DIM = 50 = as we are loading miniboones dataset(mb) by default, it has 50 features
    ACTION_DIM = FEATURE_DIM + CLASSES
    STATE_DIM = FEATURE_DIM * 2( multiply by 2 because of the mask)
              = First two coloum of the SATE_DIM specify classes(assuming miniboone dataset is loaded by deffault)
    MAX_MASK_CONST = int(1e6) = for not considering the actions that are already performed
"""
#%%
import numpy as np
from env import Environment
from IPython.core.debugger import set_trace
from consts import *
from ipdb import set_trace

class Agent():
    def __init__(self, env, pool, brain):
        self.env  = env
        self.pool = pool
        self.brain = brain

        self.epsilon = EPSILON_START
        self.s = self.env.reset()  # initialize state to zeros.

    def store(self, x):
        """
        Params:
            x = sample expreience = (s, a, s', r)
        """
        self.pool.put(x)

    def act(self, s):
        """
        Params:
            s = batch of input states(dimension = FEATURE_DIM *2)
        Outputs:
            a = batch of actions
        """
        m = np.zeros((AGENTS, ACTION_DIM))    # create max_mask
        m[:, CLASSES:] = s[:, FEATURE_DIM:]

        if self.epsilon < 1.0:
            p = self.brain.predict_np(s) - MAX_MASK_CONST * m     # select an action not considering those  already performed
            a = np.argmax(p, axis=1)
        else:
            a = np.zeros(AGENTS, dtype=np.int32)

        # override with random action
        rand_agents = np.where( np.random.rand(AGENTS) < self.epsilon )[0] #returns a touple , [0] = first index of the touple
        rand_number = np.random.rand(len(rand_agents))

        for i in range(len(rand_agents)):
            agent = rand_agents[i]

            possible_actions = np.where( m[agent] == 0. )[0]     # select a random action, don't repeat an action
            w = int(rand_number[i] * len(possible_actions))
            a[agent] = possible_actions[w]

        return a

    def step(self):
        a = self.act(self.s)
        s_, r = self.env.step(a)

        self.store( (self.s, a, r, s_) )

        self.s = s_

    def update_epsilon(self, epoch):
        if epoch >= EPSILON_EPOCHS:
            self.epsilon = EPSILON_END
        else:
            self.epsilon = EPSILON_START + epoch * (EPSILON_END - EPSILON_START) / EPSILON_EPOCHS