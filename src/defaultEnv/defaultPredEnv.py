import os
import random

import gym
import tensorflow as tf
import numpy as np
import pandas as pd
from gym import spaces, logger
import json

class defaultEnv(gym.Env):
    """
        Environment for default prediction of the company. This will take data of only one company.  
    """
    def __init__(self, data):
        super(defaultEnv, self).__init__()
        self.data = data
        
        self.REWARD_RANGE = (0, 10e5)
        self.n_actions = 2
        
        # Descrte action space.
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Observation space.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,6), dtype=np.float16)
        
    def _next_observation(self):
        """
            Return the next observation space as array
        """
        # frame = np.array([]) 
        raise NotImplementedError
    
    def _take_action(self, action):
        """
            Takes the action using the action space provided.
        """
        action_type = action[0]
        return None
        
    def step(self, action):
        """
            Take one step in the environment.
        """
        raise NotImplementedError
    
    def reset(self):
        """
            Reset the environment.
        """
        self.current_step = random.randint(
            0, len(self.data.loc[:, 'Open'].values) - 6)
        return self._next_observation()
    
    def render(self, mode='human', close=False):
        """
            Render the environment.
        """
        raise NotImplementedError
    
    def close(self):
        raise SystemExit("Stopping the environment gracefully...")