import gym
import numpy as np

#from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open import SawyerDrawerOpenEnv
from multiworld.core.wrapper_env import ProxyEnv
from multiworld.core.serializable import Serializable

import threading

from mujoco_py import MjRenderContext
import metaworld.envs.mujoco.sawyer_xyz as sawyer

class MetaWorldDrawerOpenEnv(ProxyEnv):
    LOCK = threading.Lock()


    def __init__(self,
                 env_name="DrawerOpen",
                 sparse_reward=False,
                 two_dimensional=False):
        self.quick_init(locals())

        with self.LOCK:
            if env_name == "DrawerOpen":
                self._env = sawyer.SawyerDrawerOpenEnv(sparse_reward=sparse_reward,
                                                       two_dimensional=two_dimensional)
            elif env_name == "DrawerClose":
                self._env = sawyer.SawyerDrawerOpenEnv(sparse_reward=sparse_reward,
                                                       two_dimensional=two_dimensional)
            else:
                raise NotImplementedError(env_name)

        super(MetaWorldDrawerOpenEnv, self).__init__(self._env)
        self._size = (48, 48)

        self._offscreen = MjRenderContext(self._env.sim, True, 0, 'egl', True)
        self._offscreen.cam.azimuth = 205
        self._offscreen.cam.elevation = -165
        self._offscreen.cam.distance = 2.0
        self._offscreen.cam.lookat[0] = 1.1
        self._offscreen.cam.lookat[1] = 1.1
        self._offscreen.cam.lookat[2] = -0.1


        self._offscreen.cam.azimuth = -30#0#-30
        self._offscreen.cam.elevation = -135#-180#-135
        self._offscreen.cam.distance = 0.5#0.4#.80
        self._offscreen.cam.lookat[0] = 0.0
        self._offscreen.cam.lookat[1] = 0.6#0.7#0.6
        self._offscreen.cam.lookat[2] = 0.2


#        self._offscreen.cam.distance = 0.25
#        self._offscreen.cam.lookat[0] = -.2
#        self._offscreen.cam.lookat[1] = 0.55
#        self._offscreen.cam.lookat[2] = 0.6
#        self._offscreen.cam.elevation = -60
#        self._offscreen.cam.azimuth = 360
#        self._offscreen.cam.trackbodyid = -1


    @property
    def observation_space(self):
        shape = self._size[0] * self._size[1] * 3
        space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape,), dtype=np.float32)
        #return gym.spaces.Dict({'image': space})
        return space


    @property
    def action_space(self):
        return self._env.action_space

    def close(self):
        return self._env.close()

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        obs = self._get_obs(state)
        return obs, reward, done, info

    def reset(self):
        with self.LOCK:
            state = self._env.reset()
        return self._get_obs(state)

    def render(self, mode):
        return self._env.render(mode)

    def _get_obs(self, state):
        self._offscreen.render(self._size[0], self._size[1], -1)
        image = np.flip(self._offscreen.read_pixels(self._size[0], self._size[1])[0], 1)
        image = image / 255.0
        #return {'image': image, 'state': state}
#        return {'image': image.reshape(-1)}
        return image.reshape(-1)
