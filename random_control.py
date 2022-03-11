#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 19:04:57 2022

@author: vittoriogiammarino
"""

import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper, ActionBonus
from gym_minigrid.window import Window
import pickle
import os

# %%

Buffer = {}

states = []
next_states = [] 
actions = []
rewards = []
terminals = []

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()
    states.append(obs)
    r = 0
    
    return r

def step(r, action):
    obs, reward, done, info = env.step(action)
    
    actions.append(action)
    next_states.append(obs)
    rewards.append(reward)
    terminals.append(done)
    
    r+=reward
    
    if done:
        print('done!')
        print('step=%s, reward=%.2f' % (env.step_count, r))
        r = reset()
    else:
        states.append(obs)
        
    return r

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-Empty-16x16-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--data_set_size",
    type=int,
    help="size at which to render tiles",
    default=int(1e5)
)


args = parser.parse_args()
env = gym.make(args.env)
env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env)

r = reset()
for t in range(args.data_set_size):
    action = env.action_space.sample()
    r = step(r, action)

Buffer["observations"] = np.array(states)
Buffer["next_observations"] = np.array(next_states)
Buffer["actions"] = np.array(actions)
Buffer["rewards"] = np.array(rewards)
Buffer["terminals"] = np.array(terminals)

if not os.path.exists("./offline_data_set"):
    os.makedirs("./offline_data_set")

a_file = open(f"./offline_data_set/data_set_{args.env}_random.pkl", "wb")
pickle.dump(Buffer, a_file)
a_file.close()
