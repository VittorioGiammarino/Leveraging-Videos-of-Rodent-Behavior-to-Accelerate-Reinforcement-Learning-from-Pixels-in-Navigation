#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 18:05:14 2022

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

def change_color(obs):
    dim = obs.shape
    obs_changed = np.array([obs[:,:,1], obs[:,:,2], obs[:,:,0]]).transpose(1,2,0)
    for i in range(dim[0]):
        for j in range(dim[1]):
                if obs_changed[i,j,0] == 0 and obs_changed[i,j,1] == 0 and obs_changed[i,j,2] == 0:
                    obs_changed[i,j,0] = 255
                    obs_changed[i,j,1] = 255
                    obs_changed[i,j,2] = 255
                    
    return obs_changed
    

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if args.change_color:
        obs = change_color(obs)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)
    states.append(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    
    if args.change_color:
        obs = change_color(obs)
    
    actions.append(action)
    next_states.append(obs)
    rewards.append(reward)
    terminals.append(done)
    
    if done:
        print('done!')
        reset()
    else:
        redraw(obs)
        states.append(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default="MiniGrid-Empty-16x16-v0"
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

parser.add_argument('--change_color', 
                    default = False,
                    action = 'store_true',
                    help = "change color of the scene")

args = parser.parse_args()
env = gym.make(args.env)
env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)
reset()

# Blocking event loop
window.show(block=True)

Buffer["observations"] = np.array(states)
Buffer["next_observations"] = np.array(next_states)
Buffer["actions"] = np.array(actions)
Buffer["rewards"] = np.array(rewards)
Buffer["terminals"] = np.array(terminals)

if not os.path.exists("./offline_data_set"):
    os.makedirs("./offline_data_set")

if args.change_color:
    a_file = open(f"./offline_data_set/data_set_modified_{args.env}_human_expert.pkl", "wb")
    pickle.dump(Buffer, a_file)
    a_file.close()
else:
    a_file = open(f"./offline_data_set/data_set_{args.env}_human_expert.pkl", "wb")
    pickle.dump(Buffer, a_file)
    a_file.close()

# %%

import matplotlib.pyplot as plt

if args.change_color:
    a_file = open(f"offline_data_set/data_set_modified_{args.env}_human_expert.pkl", "rb")
    data_set_expert_mod = pickle.load(a_file)
    
else:
    a_file = open(f"offline_data_set/data_set_{args.env}_human_expert.pkl", "rb")
    data_set_expert_mod = pickle.load(a_file)
    

plt.imshow(data_set_expert_mod['observations'][100])