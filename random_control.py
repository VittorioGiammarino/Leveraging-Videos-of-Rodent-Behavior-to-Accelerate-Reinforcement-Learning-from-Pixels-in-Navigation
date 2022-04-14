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

import matplotlib.pyplot as plt

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

# %%

env = 'MiniGrid-Empty-16x16-v0'
data_set = 'random'
a_file = open(f"offline_data_set/data_set_{env}_{data_set}.pkl", "rb")
data_set_expert = pickle.load(a_file)

# %%
sample = data_set_expert['observations'][0]
sample_changed = np.array([sample[:,:,1],sample[:,:,2],sample[:,:,0]]).transpose(1,2,0)
dim = sample_changed.shape


modified_set_expert = {}
modified_set_expert['observations'] = []

for data in range(len(data_set_expert['observations'])):
    sample = data_set_expert['observations'][data]
    dim = sample.shape
    sample_changed = np.array([sample[:,:,1],sample[:,:,2],sample[:,:,0]]).transpose(1,2,0)
    for i in range(dim[0]):
        for j in range(dim[1]):
                if sample_changed[i,j,0] == 0 and sample_changed[i,j,1] == 0 and sample_changed[i,j,2] == 0:
                    sample_changed[i,j,0] = 255
                    sample_changed[i,j,1] = 255
                    sample_changed[i,j,2] = 255
                    
    modified_set_expert['observations'].append(sample_changed)
 
# %%

modified_set_expert['observations'] = np.array(modified_set_expert['observations'])
               
if not os.path.exists("./offline_data_set"):
    os.makedirs("./offline_data_set")

a_file = open(f"./offline_data_set/data_set_modified_{env}_{data_set}.pkl", "wb")
pickle.dump(modified_set_expert, a_file)
a_file.close()

# %%

a_file = open(f"offline_data_set/data_set_modified_{env}_{data_set}.pkl", "rb")
data_set_expert_mod = pickle.load(a_file)

plt.imshow(data_set_expert_mod['observations'][100])

