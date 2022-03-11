#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:02:25 2021

@author: vittorio
"""

import copy
import numpy as np
import gym
from gym.wrappers import FilterObservation, FlattenObservation
import torch
from gym_minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def FlatStochasticSampleTrajMDP(seed, policy, args):
    
    eval_env = gym.make(args.env)
    
    if args.grid_observability == 'Partial':
        eval_env = RGBImgPartialObsWrapper(eval_env)
    elif args.grid_observability == 'Fully':
        eval_env = RGBImgObsWrapper(eval_env)
    else:
        print("Special encoding Environmnet")
            
    eval_env = ImgObsWrapper(eval_env)
    
    eval_env.seed(seed + 100)
    eval_env._max_episode_steps = args.evaluation_max_n_steps
     
    Reward_array = np.empty((0,0),int)
   
    policy.actor.eval()
    
    for t in range(args.evaluation_episodes):
        state, done = eval_env.reset(), False
        
        cum_reward = 0 
        
        for _ in range(0, args.evaluation_max_n_steps):
            action = policy.select_action(state)
            
            state, reward, done, _ = eval_env.step(action)
            cum_reward = cum_reward + reward  
            
            if done:
                break
            
        Reward_array = np.append(Reward_array, cum_reward)
        
    return Reward_array  
            

def eval_policy(seed, policy, args):

    Reward = FlatStochasticSampleTrajMDP(seed, policy, args)
    avg_reward = np.sum(Reward)/args.evaluation_episodes

    print("---------------------------------------")
    print(f"Seed {seed}, Evaluation over {args.evaluation_episodes}, episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    
    return avg_reward

