import gym
from gym.envs.registration import register
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)

_REGISTERED = False

# print("\n\n\n\n\nin avi multiworld\n\n\n\n\n")

def register_goal_example_envs():
    register(
        id='BaseHumanLikeSawyerPushForwardEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, 0.0, 0.75),
            'indicator_threshold': 0.03,
            'reward_type': 'puck_success_positive',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_push_puck_to_square_human_like.xml',
            'hide_goal_markers': True,
            'puck_random_init': True,
        }
        )

    register(
        id='Image48HumanLikeSawyerPushForwardEnv-v0',
        entry_point=create_image_48_human_like_sawyer_push_forward_v0,
        )
    
    register(
        id='Image48HumanLikeSawyerPushForwardEnv-v1',
        entry_point=create_image_48_human_like_sawyer_push_forward_v1,
        )
    
    register(
        id='Image48HumanLikeSawyerPushForwardEnvDiscrete-v1',
        entry_point=create_image_48_human_like_sawyer_push_forward_Discrete_v1,
        )
    
def create_image_48_human_like_sawyer_push_forward_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseHumanLikeSawyerPushForwardEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_image_48_human_like_sawyer_push_forward_v1():
    from multiworld.core.img_goal_env import ImgGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    size = 48
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseHumanLikeSawyerPushForwardEnv-v0'),
        imsize=size,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=False,
        )
    
    return ImgGoalEnv(image_env, imsize = size, obs_keys=['image_observation'])

def create_image_48_human_like_sawyer_push_forward_Discrete_v1():
    from multiworld.core.img_goal_env import ImgGoalEnvDiscrete
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    size = 48
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseHumanLikeSawyerPushForwardEnv-v0'),
        imsize=size,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=False,
        )
    
    return ImgGoalEnvDiscrete(image_env, imsize = size, obs_keys=['image_observation'])
