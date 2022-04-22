from gym.spaces import Box, Dict
import numpy as np

from multiworld.core.wrapper_env import ProxyEnv


class ImgGoalEnvDiscrete(ProxyEnv):
    def __init__(
            self,
            wrapped_env,
            imsize = 48,
            obs_keys=None,
            goal_keys=None,
            append_goal_to_obs=False,
    ):
        self.quick_init(locals())
        super(ImgGoalEnvDiscrete, self).__init__(wrapped_env)

        self.imsize = imsize

        if obs_keys is None:
            obs_keys = ['observation']
        if goal_keys is None:
            goal_keys = ['desired_goal']
        if append_goal_to_obs:
            obs_keys += goal_keys
        for k in obs_keys:
            assert k in self.wrapped_env.observation_space.spaces
        for k in goal_keys:
            assert k in self.wrapped_env.observation_space.spaces
        assert isinstance(self.wrapped_env.observation_space, Dict)

        self.obs_keys = obs_keys
        self.goal_keys = goal_keys
        # TODO: handle nested dict
        self.observation_space = Box(
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].low
                for k in obs_keys
            ]),
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].high
                for k in obs_keys
            ]),
        )
        self.goal_space = Box(
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].low
                for k in goal_keys
            ]),
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].high
                for k in goal_keys
            ]),
        )
        self._goal = None
        
        self.step_counter = 0
        self.max_number_steps = 1024
        self.action_space.n = 5
        self.actions_array_env = [[0,1], [1,0], [0,-1], [-1,0], [0,0]]

    def step(self, action):
        self.step_counter+=1
        try:
            action = int(action.flatten())
        except:
            action = int(action)
            
        action_env = np.array(self.actions_array_env[action])
        _, reward, done, info = self.wrapped_env.step(action_env)
        
        if self.step_counter >= self.max_number_steps:
            done = True
            
        img_obs = self.wrapped_env.get_image(self.imsize, self.imsize)
        return img_obs, reward, done, info

    def reset(self):
        self.step_counter = 0
        obs = self.wrapped_env.reset()
        img_obs = self.wrapped_env.get_image(self.imsize, self.imsize)
        self._goal = np.hstack([obs[k] for k in self.goal_keys])
        return img_obs

    def get_goal(self):
        return self._goal
    
class ImgGoalEnv(ProxyEnv):
    def __init__(
            self,
            wrapped_env,
            imsize = 48,
            obs_keys=None,
            goal_keys=None,
            append_goal_to_obs=False,
    ):
        self.quick_init(locals())
        super(ImgGoalEnv, self).__init__(wrapped_env)

        self.imsize = imsize

        if obs_keys is None:
            obs_keys = ['observation']
        if goal_keys is None:
            goal_keys = ['desired_goal']
        if append_goal_to_obs:
            obs_keys += goal_keys
        for k in obs_keys:
            assert k in self.wrapped_env.observation_space.spaces
        for k in goal_keys:
            assert k in self.wrapped_env.observation_space.spaces
        assert isinstance(self.wrapped_env.observation_space, Dict)

        self.obs_keys = obs_keys
        self.goal_keys = goal_keys
        # TODO: handle nested dict
        self.observation_space = Box(
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].low
                for k in obs_keys
            ]),
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].high
                for k in obs_keys
            ]),
        )
        self.goal_space = Box(
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].low
                for k in goal_keys
            ]),
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].high
                for k in goal_keys
            ]),
        )
        self._goal = None
        
        self.step_counter = 0
        self.max_number_steps = 1024

    def step(self, action):
        self.step_counter+=1
        _, reward, done, info = self.wrapped_env.step(action)
        if self.step_counter >= self.max_number_steps:
            done = True
            
        img_obs = self.wrapped_env.get_image(self.imsize, self.imsize)
        return img_obs, reward, done, info

    def reset(self):
        self.step_counter = 0
        obs = self.wrapped_env.reset()
        img_obs = self.wrapped_env.get_image(self.imsize, self.imsize)
        self._goal = np.hstack([obs[k] for k in self.goal_keys])
        return img_obs

    def get_goal(self):
        return self._goal
