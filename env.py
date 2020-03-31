import numpy as np
from IPython.core.debugger import set_trace
from consts import *
#==============================

LIN_ARRAY = np.arange(AGENTS)
#==============================
class Environment:
    def __init__(self, data, costs, ff):
        self.data_x = data.iloc[:, 0:-1].astype('float32').values
        self.data_y = data.iloc[:,   -1].astype('int32').values
        self.data_len = len(data)

        self.costs = costs.values

        self.mask = np.zeros( (AGENTS, FEATURE_DIM) )
        self.x    = np.zeros( (AGENTS, FEATURE_DIM) )
        self.y    = np.zeros( AGENTS )

        self.ff = ff  # ff is feature cost

    def reset(self):
        """
        initialize input state to zero vector
        """
        for i in range(AGENTS):
            self._reset(i)
        return self._get_state()

    def _reset(self, i):
        """
        Initialize the mask to zero
        Randomly choose a data point and assign it to x[i] and y[i]
        """
        self.mask[i] = 0
        self.x[i], self.y[i] = self._generate_sample()

    def step(self, action):
        self.mask[LIN_ARRAY, action - CLASSES] = 1

        r = -self.costs[action - CLASSES] * self.ff

        for i in np.where(action < CLASSES)[0]:
            r[i] = REWARD_CORRECT if action[i] == self.y[i] else REWARD_INCORRECT
            self._reset(i)

        s_ = self._get_state()

        return (s_, r)

    def _generate_sample(self):
        """
        randomly select a datapoint(x,y)
        """
        idx = np.random.randint(0, self.data_len)

        x = self.data_x[idx] # randomly chosen data sample
        y = self.data_y[idx]

        return (x, y)

    def _get_state(self):
        """
        Make a input state
        """
        x_ = self.x * self.mask # can see points that has mask one
        x_ = np.concatenate( (x_, self.mask), axis=1 ).astype(np.float32) # input
        return x_
