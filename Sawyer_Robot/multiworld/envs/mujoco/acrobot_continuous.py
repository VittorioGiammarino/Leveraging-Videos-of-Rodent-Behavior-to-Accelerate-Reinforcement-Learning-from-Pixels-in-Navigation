import gym
from gym import spaces
import numpy as np
from gym.envs.classic_control.acrobot import AcrobotEnv

class AcrobotContinuousEnv(AcrobotEnv):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

    def step(self, action):
        discrete_action = 1
        if action < -0.5:
            discrete_action = 0
        elif action > 0.5:
            discrete_action = 2
        return super().step(discrete_action)
