#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:17:27 2021

@author: vittorio
"""
import torch
import argparse
import os
import numpy as np
import gym 
import pickle
from gym_minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper, ActionBonus

from Buffers.vanilla_buffer import ReplayBuffer

import runner

from algorithms.RL.SAC import SAC
from algorithms.RL.AWAC import AWAC
from algorithms.RL.TD3 import TD3
from algorithms.RL.AWAC_GAE import AWAC_GAE
from algorithms.RL.AWAC_Q_lambda import AWAC_Q_lambda
from algorithms.RL.PPO import PPO
from algorithms.RL.GePPO import GePPO

from algorithms.offline_RL.TD3_BC import TD3_BC
from algorithms.offline_RL.SAC_BC import SAC_BC
from algorithms.offline_RL.AWAC_off import AWAC_off
from algorithms.offline_RL.BC import BC
from algorithms.offline_RL.AWAC_Q_lambda_off import AWAC_Q_lambda_off
from algorithms.offline_RL.AWR_off import AWR_off

from algorithms.on_off_RL_demonstrations.on_off_SAC import on_off_SAC
from algorithms.on_off_RL_demonstrations.on_off_AWAC import on_off_AWAC
from algorithms.on_off_RL_demonstrations.on_off_AWAC_Q_lambda_Peng import on_off_AWAC_Q_lambda_Peng
from algorithms.on_off_RL_demonstrations.on_off_AWAC_Q_lambda_Haru import on_off_AWAC_Q_lambda_Haru
from algorithms.on_off_RL_demonstrations.on_off_AWAC_TB import on_off_AWAC_TB
from algorithms.on_off_RL_demonstrations.on_off_AWAC_GAE import on_off_AWAC_GAE

from algorithms.on_off_RL_observations.on_off_SAC_obs import on_off_SAC_obs
from algorithms.on_off_RL_observations.on_off_AWAC_obs import on_off_AWAC_obs
from algorithms.on_off_RL_observations.on_off_AWAC_Q_lambda_Peng_obs import on_off_AWAC_Q_lambda_Peng_obs
from algorithms.on_off_RL_observations.on_off_AWAC_Q_lambda_Haru_obs import on_off_AWAC_Q_lambda_Haru_obs
from algorithms.on_off_RL_observations.on_off_AWAC_TB_obs import on_off_AWAC_TB_obs
from algorithms.on_off_RL_observations.on_off_AWAC_GAE_obs import on_off_AWAC_GAE_obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
    
def RL(env, args, seed):
    
    if args.action_space == 'Continuous':
        action_dim = env.action_space.shape[0] 
        action_space_cardinality = np.inf
        max_action = np.zeros((action_dim,))
        min_action = np.zeros((action_dim,))
        for a in range(action_dim):
            max_action[a] = env.action_space.high[a]   
            min_action[a] = env.action_space.low[a]  
            
    elif args.action_space == 'Discrete':
        try:
            action_dim = env.action_space.shape[0] 
        except:
            action_dim = 1

        action_space_cardinality = env.action_space.n
        max_action = np.nan
        min_action = np.nan
                
    state_dim = env.reset().shape
    
    #Buffers
    replay_buffer = ReplayBuffer(args.action_space, state_dim, action_dim)
    if args.mode == "offline_RL":
        
        a_file = open(f"offline_data_set/data_set_{args.env}_{args.data_set}.pkl", "rb")
        data_set = pickle.load(a_file)
        
        replay_buffer = ReplayBuffer(args.action_space, state_dim, action_dim, len(data_set["observations"])-1)
        replay_buffer.convert_D4RL(data_set) 
        
        if args.policy == "TD3":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "BC": args.BC
            }
    
            Agent_RL = TD3_BC(**kwargs)
            
            run_sim = runner.run_TD3_BC(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "SAC":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "BC": args.BC
            }
    
            Agent_RL = SAC_BC(**kwargs)
            
            run_sim = runner.run_SAC_BC(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "AWAC":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy
            }
    
            Agent_RL = AWAC_off(**kwargs)
            
            run_sim = runner.run_AWAC_off(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "BC":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action
            }
    
            Agent_RL = BC(**kwargs)
            
            run_sim = runner.run_SAC_BC(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "AWAC_Q_lambda_Peng":
            kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "action_space_cardinality": action_space_cardinality,
                "max_action": max_action,
                "min_action": min_action,
                "Entropy": args.Entropy
                }
    
            Agent_RL = AWAC_Q_lambda_off(**kwargs)
            
            run_sim = runner.run_AWAC_Q_lambda_off_Peng(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "AWAC_Q_lambda_Haru":
            kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "action_space_cardinality": action_space_cardinality,
                "max_action": max_action,
                "min_action": min_action,
                "Entropy": args.Entropy
                }
    
            Agent_RL = AWAC_Q_lambda_off(**kwargs)
            
            run_sim = runner.run_AWAC_Q_lambda_off_Haru(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL 
        
        if args.policy == "AWAC_Q_lambda_TB":
            kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "action_space_cardinality": action_space_cardinality,
                "max_action": max_action,
                "min_action": min_action,
                "Entropy": args.Entropy
                }
    
            Agent_RL = AWAC_Q_lambda_off(**kwargs)
            
            run_sim = runner.run_AWAC_Q_lambda_off_TB(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL 
        
        if args.policy == "AWR":
            kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "action_space_cardinality": action_space_cardinality,
                "max_action": max_action,
                "min_action": min_action,
                "Entropy": args.Entropy
                }
    
            Agent_RL = AWR_off(**kwargs)
            
            run_sim = runner.run_AWR_off(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL 
        
    elif args.mode == "on_off_RL_from_demonstrations":
        
        replay_buffer_online = ReplayBuffer(args.action_space, state_dim, action_dim)
        
        a_file = open(f"offline_data_set/data_set_{args.env}_{args.data_set}.pkl", "rb")
        data_set = pickle.load(a_file)
        
        replay_buffer_offline = ReplayBuffer(args.action_space, state_dim, action_dim, len(data_set["observations"])-1)
        replay_buffer_offline.convert_D4RL(data_set) 
        
        if args.policy == "SAC":            
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action
            }
    
            Agent_RL = on_off_SAC(**kwargs)
            
            run_sim = runner.run_on_off_SAC(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer_online, replay_buffer_offline, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL
        
        if args.policy == "AWAC":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
            }
    
            Agent_RL = on_off_AWAC(**kwargs)
            
            run_sim = runner.run_on_off_AWAC(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer_online, replay_buffer_offline, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL
        
        if args.policy == "AWAC_Q_lambda_Peng":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = on_off_AWAC_Q_lambda_Peng(**kwargs)
            
            run_sim = runner.run_on_off_AWAC_Q_lambda_Peng(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer_offline, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "AWAC_Q_lambda_Haru":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = on_off_AWAC_Q_lambda_Haru(**kwargs)
            
            run_sim = runner.run_on_off_AWAC_Q_lambda_haru(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer_offline, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "AWAC_Q_lambda_TB":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = on_off_AWAC_TB(**kwargs)
            
            run_sim = runner.run_on_off_AWAC_TB(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer_offline, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "AWAC_GAE":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = on_off_AWAC_GAE(**kwargs)
            
            run_sim = runner.run_on_off_AWAC_GAE(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer_offline, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
    
    elif args.mode == "on_off_RL_from_observations":
        
        replay_buffer_online = ReplayBuffer(args.action_space, state_dim, action_dim)
        
        if args.data_set == 'rodent':
            assert args.env == "MiniGrid-Empty-16x16-v0"
            with open('data_set/rodent_data_processed.npy', 'rb') as f:
                off_policy_observations = np.load(f, allow_pickle=True)
                
        elif args.data_set == 'modified_human_expert':
            assert args.env == "MiniGrid-Empty-16x16-v0" or args.env == "MiniGrid-FourRooms-v0"
            
            a_file = open(f"offline_data_set/data_set_modified_{args.env}_human_expert.pkl", "rb")
            data_set = pickle.load(a_file)
            off_policy_observations = data_set['observations'].transpose(0,3,1,2)
            
        elif args.data_set == 'human_expert':
            assert args.env == "MiniGrid-Empty-16x16-v0"
            a_file = open(f"offline_data_set/data_set_{args.env}_{args.data_set}.pkl", "rb")
            data_set = pickle.load(a_file)
            off_policy_observations = data_set['observations'].transpose(0,3,1,2)
                
        else:
            NotImplemented
            
        if args.policy == "SAC":            
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Domain_adaptation": args.domain_adaptation,
             "intrinsic_reward": args.intrinsic_reward
            }
    
            Agent_RL = on_off_SAC_obs(**kwargs)
            
            run_sim = runner.run_on_off_SAC(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer_online, off_policy_observations, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL
        
        if args.policy == "AWAC":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Domain_adaptation": args.domain_adaptation,
             "Entropy": args.Entropy,
             "intrinsic_reward": args.intrinsic_reward
            }
    
            Agent_RL = on_off_AWAC_obs(**kwargs)
            
            run_sim = runner.run_on_off_AWAC(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer_online, off_policy_observations, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL
        
        if args.policy == "AWAC_Q_lambda_Peng":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Domain_adaptation": args.domain_adaptation,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter,
             "intrinsic_reward": args.intrinsic_reward,
             "number_obs_off_per_traj": args.number_obs_per_traj
            }
    
            Agent_RL = on_off_AWAC_Q_lambda_Peng_obs(**kwargs)
            
            run_sim = runner.run_on_off_AWAC_Q_lambda_Peng(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, off_policy_observations, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL 
        
        if args.policy == "AWAC_Q_lambda_Haru":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Domain_adaptation": args.domain_adaptation,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter,
             "intrinsic_reward": args.intrinsic_reward,
             "number_obs_off_per_traj": args.number_obs_per_traj
            }
    
            Agent_RL = on_off_AWAC_Q_lambda_Haru_obs(**kwargs)
            
            run_sim = runner.run_on_off_AWAC_Q_lambda_Haru(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, off_policy_observations, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "AWAC_Q_lambda_TB":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Domain_adaptation": args.domain_adaptation,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter,
             "intrinsic_reward": args.intrinsic_reward,
             "number_obs_off_per_traj": args.number_obs_per_traj
            }
    
            Agent_RL = on_off_AWAC_TB_obs(**kwargs)
            
            run_sim = runner.run_on_off_AWAC_TB(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, off_policy_observations, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "AWAC_GAE":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Domain_adaptation": args.domain_adaptation,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter,
             "intrinsic_reward": args.intrinsic_reward,
             "number_obs_off_per_traj": args.number_obs_per_traj
            }
    
            Agent_RL = on_off_AWAC_GAE_obs(**kwargs)
            
            run_sim = runner.run_on_off_AWAC_GAE(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, off_policy_observations, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
            
    
    elif args.mode == "RL":
        
        if args.policy == "SAC":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action
            }
    
            Agent_RL = SAC(**kwargs)
            
            run_sim = runner.run_SAC(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL
        
        if args.policy == "AWAC":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
            }
    
            Agent_RL = AWAC(**kwargs)
            
            run_sim = runner.run_AWAC(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL
        
        if args.policy == "TD3":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action
            }
    
            Agent_RL = TD3(**kwargs)
            
            run_sim = runner.run_TD3(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL 
        
        if args.policy == "AWAC_GAE":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = AWAC_GAE(**kwargs)
            
            run_sim = runner.run_AWAC_GAE(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "AWAC_Q_lambda_Peng":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = AWAC_Q_lambda(**kwargs)
            
            run_sim = runner.run_AWAC_Q_lambda_Peng(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "AWAC_Q_lambda_Haru":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = AWAC_Q_lambda(**kwargs)
            
            run_sim = runner.run_AWAC_Q_lambda_Haru(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL 
        
        if args.policy == "AWAC_Q_lambda_TB":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = AWAC_Q_lambda(**kwargs)
            
            run_sim = runner.run_AWAC_Q_lambda_TB(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL 
    
        if args.policy == "PPO":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = PPO(**kwargs)
            
            run_sim = runner.run_PPO(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
                
        if args.policy == "GePPO":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter,
            }
    
            Agent_RL = GePPO(**kwargs)
            
            run_sim = runner.run_GePPO(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL 
        
def train(args, seed): 
    
    env = gym.make(args.env)
    
    if args.grid_observability == 'Partial':
        env = RGBImgPartialObsWrapper(env)
    elif args.grid_observability == 'Fully':
        
        if args.env == "MiniGrid-Empty-32x32-v0":
            env = RGBImgObsWrapper(env, tile_size=4)
        else:
            env = RGBImgObsWrapper(env)
    else:
        print("Special encoding Environmnet")
        
    if args.exploration_bonus:
        print("Exploration Bonus True")
        env = ActionBonus(env)
            
    env = ImgObsWrapper(env)
    
    try:
        if env.action_space.n>0:
            args.action_space = "Discrete"
            print("Environment supports Discrete action space.")
    except:
        args.action_space = "Continuous"
        print("Environment supports Continuous action space.")
            
    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    wallclock_time, evaluations, policy = RL(env, args, seed)
    
    return wallclock_time, evaluations, policy


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    #General
    parser.add_argument("--mode", default="on_off_RL_from_observations", help='RL, offline_RL, on_off_RL_from_demonstrations, on_off_RL_from_observations')     
    parser.add_argument("--env", default="MiniGrid-Empty-32x32-v0", help = 'MiniGrid-Empty-16x16-v0 or MiniGrid-Empty-32x32-v0')  
    parser.add_argument("--data_set", default="rodent", help="random, human_expert, rodent, modified_human_expert")  
    parser.add_argument("--action_space", default="Discrete")  # Discrete or continuous
    parser.add_argument("--grid_observability", default="Fully", help="Partial or Fully observable")
    parser.add_argument("--exploration_bonus", action = "store_true", help="reward to encourage exploration of less visited (state,action) pairs")
    
    parser.add_argument("--policy", default="AWAC_GAE") 
    parser.add_argument("--seed", default=10, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--number_steps_per_iter", default=4*32*32*4, type=int) # Number of steps between two evaluations (default Minigrid: 4096)
    parser.add_argument("--eval_freq", default=1, type=int)          # How many iterations we evaluate
    parser.add_argument("--max_iter", default=50, type=int)    # Max number of iterations to run environment, max_steps = max_iter*number_steps_per_iter
    # RL
    parser.add_argument("--start_timesteps", default=5e3, type=int) # Time steps before training default=5e3 (default Minigrid: 5000, default Sawyer: 25000)
    parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise    
    # offline RL
    parser.add_argument("--ntrajs", default=5, type=int) #default: 10, number of off-policy trajectories 
    parser.add_argument("--number_obs_per_traj", default=100, type=int) # number of off-policy demonstrations or observations used for training at each iteration (default Minigrid: 100, default Sawyer: 500)
    parser.add_argument("--Entropy", action="store_true")
    parser.add_argument("--BC", action="store_true")
    # from observations
    parser.add_argument("--domain_adaptation", action="store_true")
    parser.add_argument("--intrinsic_reward", default=0.01, type=float) #0.01 or 0.005 or 0
    # Evaluation
    parser.add_argument("--evaluation_episodes", default=10, type=int) #default: 10, number of episodes per evaluation
    parser.add_argument("--evaluation_max_n_steps", default = 4*32*32, type=int) #default: 2000, max number of steps evaluation episode
    # Experiments
    parser.add_argument("--detect_gradient_anomaly", action="store_true")
    args = parser.parse_args()
    
    torch.autograd.set_detect_anomaly(args.detect_gradient_anomaly)
    
    assert args.env == "MiniGrid-Empty-16x16-v0" or args.env == "MiniGrid-Empty-32x32-v0" or args.env == "MiniGrid-FourRooms-v0"
        
    if args.env == "MiniGrid-Empty-16x16-v0" or args.env == "MiniGrid-Empty-32x32-v0":
        args.reward_given = False
        
    if args.mode == "RL":
        
        if args.policy == "TD3" or args.policy == "SAC":
            file_name = f"{args.mode}_{args.policy}_{args.env}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
            print("---------------------------------------")
        else:
            file_name = f"{args.mode}_{args.policy}_Entropy_{args.Entropy}_{args.env}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}_Entropy_{args.Entropy}, Env: {args.env}, Seed: {args.seed}")
            print("---------------------------------------")
        
        if not os.path.exists(f"./results/{args.mode}"):
            os.makedirs(f"./results/{args.mode}")
            
        if not os.path.exists(f"./Saved_models/{args.mode}/{file_name}"):
            os.makedirs(f"./Saved_models/{args.mode}/{file_name}")
        
        wallclock_time, evaluations, policy = train(args, args.seed)
        
        np.save(f"./results/{args.mode}/evaluation_{file_name}", evaluations)
        np.save(f"./results/{args.mode}/wallclock_time_{file_name}", wallclock_time)
        policy.save_actor(f"./Saved_models/{args.mode}/{file_name}/{file_name}")
            
    if args.mode == "offline_RL":
        
        if args.policy == "AWAC_Q_lambda_TB" or args.policy == "AWAC_Q_lambda_Peng" or args.policy == "AWAC_Q_lambda_Haru" or args.policy == "AWAC" or args.policy == "AWR":
            file_name = f"{args.mode}_{args.policy}_Entropy_{args.Entropy}_{args.env}_dataset_{args.data_set}_{args.seed}"
            print("--------------------------------------------------------------------------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}_Entropy_{args.Entropy}, Env: {args.env}, Data: {args.data_set}, Seed: {args.seed}")
            print("--------------------------------------------------------------------------------------------------------")
                      
        else:
            if args.BC:
                file_name = f"{args.mode}_{args.policy}_BC_{args.env}_dataset_{args.data_set}_{args.seed}"
                print("---------------------------------------")
                print(f"Mode: {args.mode}, Policy: {args.policy}_BC, Env: {args.env}, Data: {args.data_set}, Seed: {args.seed}")
                print("---------------------------------------")
                
            else:
                file_name = f"{args.mode}_{args.policy}_{args.env}_dataset_{args.data_set}_{args.seed}"
                print("---------------------------------------")
                print(f"Mode: {args.mode}, Policy: {args.policy}, Env: {args.env}, Data: {args.data_set}, Seed: {args.seed}")
                print("---------------------------------------")
         
        if not os.path.exists(f"./results/{args.mode}"):
            os.makedirs(f"./results/{args.mode}")
            
        if not os.path.exists(f"./Saved_models/{args.mode}/{file_name}"):
            os.makedirs(f"./Saved_models/{args.mode}/{file_name}")
        
        wallclock_time, evaluations, policy = train(args, args.seed)
        
        np.save(f"./results/{args.mode}/evaluation_{file_name}", evaluations)
        np.save(f"./results/{args.mode}/wallclock_time_time_{file_name}", wallclock_time)
        policy.save_actor(f"./Saved_models/{args.mode}/{file_name}/{file_name}")
            
    if args.mode == "on_off_RL_from_demonstrations":
        
        if args.policy == "SAC":
            file_name = f"{args.mode}_{args.policy}_{args.env}_dataset_{args.data_set}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}, Env: {args.env}, Data: {args.data_set}, Seed: {args.seed}")
            print("---------------------------------------")
        else:
            file_name = f"{args.mode}_{args.policy}_Entropy_{args.Entropy}_{args.env}_dataset_{args.data_set}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}_Entropy_{args.Entropy}, Env: {args.env}, Data: {args.data_set}, Seed: {args.seed}")
            print("---------------------------------------")
        
        if not os.path.exists(f"./results/{args.mode}"):
            os.makedirs(f"./results/{args.mode}")
            
        if not os.path.exists(f"./Saved_models/{args.mode}/{file_name}"):
            os.makedirs(f"./Saved_models/{args.mode}/{file_name}")
        
        wallclock_time, evaluations, policy = train(args, args.seed)
        
        np.save(f"./results/{args.mode}/evaluation_{file_name}", evaluations)
        np.save(f"./results/{args.mode}/wallclock_time_{file_name}", wallclock_time)
        policy.save_actor(f"./Saved_models/{args.mode}/{file_name}/{file_name}")
            
    if args.mode == "on_off_RL_from_observations":
        
        if args.policy == "SAC":
            file_name = f"{args.mode}_{args.policy}_{args.env}_dataset_{args.data_set}_domain_adaptation_{args.domain_adaptation}_ri_{args.intrinsic_reward}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}, Env: {args.env}, Data: {args.data_set}, Domain Adaptation: {args.domain_adaptation}, ri: {args.intrinsic_reward}, Seed: {args.seed}")
            print("---------------------------------------")
        else:
            file_name = f"{args.mode}_{args.policy}_Entropy_{args.Entropy}_{args.env}_dataset_{args.data_set}_domain_adaptation_{args.domain_adaptation}_ri_{args.intrinsic_reward}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}_Entropy_{args.Entropy}, Env: {args.env}, Data: {args.data_set}, Domain Adaptation: {args.domain_adaptation}, ri: {args.intrinsic_reward}, Seed: {args.seed}")
            print("---------------------------------------")
        
        if not os.path.exists(f"./results/{args.mode}"):
            os.makedirs(f"./results/{args.mode}")
            
        if not os.path.exists(f"./Saved_models/{args.mode}/{file_name}"):
            os.makedirs(f"./Saved_models/{args.mode}/{file_name}")
        
        wallclock_time, evaluations, policy = train(args, args.seed)
        
        np.save(f"./results/{args.mode}/evaluation_{file_name}", evaluations)
        np.save(f"./results/{args.mode}/wallclock_time_{file_name}", wallclock_time)
        policy.save_actor(f"./Saved_models/{args.mode}/{file_name}/{file_name}")

        
        
    
   
                
                
                
