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

from algorithms.SAC import SAC
from algorithms.TD3 import TD3

from algorithms.TD3_BC import TD3_BC
from algorithms.SAC_BC import SAC_BC
from algorithms.BC import BC
from algorithms.PPO_off import PPO_off

from algorithms.PPO import PPO
from algorithms.Vanilla_A2C import Vanilla_A2C
from algorithms.A2C import A2C
from algorithms.GePPO import GePPO
from algorithms.GeA2C import GeA2C

# import PPO_from_videos

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
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    if args.mode == "offline_RL":
        
        a_file = open(f"offline_data_set/data_set_{args.env}_{args.data_set}.pkl", "rb")
        data_set = pickle.load(a_file)
        
        replay_buffer = ReplayBuffer(state_dim, action_dim, len(data_set["observations"])-1)
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
        
        if args.policy == "PPO_off":
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "BC": args.BC,
             "Entropy": args.Entropy,
             "GAE": args.GAE
            }
    
            Agent_RL = PPO_off(**kwargs)
            
            run_sim = runner.run_PPO_off(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
    elif args.mode == "off-on-RL":
        with open('data_set/rodent_data_processed.npy', 'rb') as f:
            External_data_set = np.load(f, allow_pickle=True)
            
        if args.policy == "PPO_from_videos":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "external_data_set" : External_data_set,
             "number_obs_per_iter": args.number_obs_per_iter,
             "num_steps_per_rollout": args.number_steps_per_iter,
            }
    
            Agent_RL = PPO_from_videos.PPO_from_videos(**kwargs)
            
            run_sim = runner.run_PPO(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
            
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
        
        if args.policy == "Vanilla_A2C":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter,
            }
    
            Agent_RL = Vanilla_A2C(**kwargs)
            
            run_sim = runner.run_Vanilla_A2C(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL  
        
        if args.policy == "A2C":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = A2C(**kwargs)
            
            run_sim = runner.run_A2C(Agent_RL)
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
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = GePPO(**kwargs)
            
            run_sim = runner.run_GePPO(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL 
        
        if args.policy == "GeA2C":        
            kwargs = {
             "state_dim": state_dim,
             "action_dim": action_dim,
             "action_space_cardinality": action_space_cardinality,
             "max_action": max_action,
             "min_action": min_action,
             "Entropy": args.Entropy,
             "num_steps_per_rollout": args.number_steps_per_iter
            }
    
            Agent_RL = GeA2C(**kwargs)
            
            run_sim = runner.run_GeA2C(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL 
    
def train(args, seed): 
    
    env = gym.make(args.env)
    
    if args.grid_observability == 'Partial':
        env = RGBImgPartialObsWrapper(env)
    elif args.grid_observability == 'Fully':
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
    parser.add_argument("--mode", default="RL", help='supported modes are HVI, HRL and RL (default = "HVI")')     
    parser.add_argument("--env", default="MiniGrid-Empty-16x16-v0")  
    parser.add_argument("--data_set", default="random", help="random or human_expert")  
    parser.add_argument("--action_space", default="Discrete")  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--grid_observability", default="Fully", help="Partial or Fully observable")
    parser.add_argument("--exploration_bonus", action = "store_true", help="reward to encourage exploration of less visited (state,action) pairs")
    
    parser.add_argument("--policy", default="PPO") # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--number_steps_per_iter", default=4096, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--number_obs_per_iter", default=2000, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--eval_freq", default=1, type=int)          # How often (time steps) we evaluate
    parser.add_argument("--max_iter", default=300, type=int)    # Max time steps to run environment
    # RL
    parser.add_argument("--start_timesteps", default=5e3, type=int) # Time steps before training default=5e3
    parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise    
    parser.add_argument("--save_model", action="store_false")         # Save model and optimizer parameters
    parser.add_argument("--load_model", default=True, type=bool)               # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model_path", default="") 
    # PPO_off
    parser.add_argument("--ntrajs", default=10, type=int)
    parser.add_argument("--GAE", action="store_true")
    parser.add_argument("--Entropy", action="store_true")
    parser.add_argument("--BC", action="store_true")
    # Evaluation
    parser.add_argument("--evaluation_episodes", default=10, type=int) #10
    parser.add_argument("--evaluation_max_n_steps", default = 2000, type=int)
    # Experiments
    parser.add_argument("--detect_gradient_anomaly", action="store_true")
    args = parser.parse_args()
    
    torch.autograd.set_detect_anomaly(args.detect_gradient_anomaly)
        
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
        
        if args.save_model: 
            np.save(f"./results/{args.mode}/evaluation_{file_name}", evaluations)
            np.save(f"./results/{args.mode}/wallclock_time_{file_name}", wallclock_time)
            policy.save_actor(f"./Saved_models/{args.mode}/{file_name}/{file_name}")
            
    if args.mode == "offline_RL":
        
        if args.policy == "PPO_off":
            
            if args.BC:
                if args.GAE:
                    file_name = f"{args.mode}_{args.policy}_BC_GAE_Entropy_{args.Entropy}_{args.env}_dataset_{args.data_set}_{args.seed}"
                    print("--------------------------------------------------------------------------------------------------------")
                    print(f"Mode: {args.mode}, Policy: {args.policy}_BC_GAE_Entropy_{args.Entropy}, Env: {args.env}, Data: {args.data_set}, Seed: {args.seed}")
                    print("--------------------------------------------------------------------------------------------------------")
                    
                else:
                    file_name = f"{args.mode}_{args.policy}_BC_TB_Entropy_{args.Entropy}_{args.env}_dataset_{args.data_set}_{args.seed}"
                    print("---------------------------------------")
                    print(f"Mode: {args.mode}, Policy: {args.policy}_BC_TB_Entropy_{args.Entropy}, Env: {args.env}, Data: {args.data_set}, Seed: {args.seed}")
                    print("---------------------------------------")
                
            else:
                if args.GAE:
                    file_name = f"{args.mode}_{args.policy}_GAE_Entropy_{args.Entropy}_{args.env}_dataset_{args.data_set}_{args.seed}"
                    print("---------------------------------------")
                    print(f"Mode: {args.mode}, Policy: {args.policy}_GAE_Entropy_{args.Entropy}, Env: {args.env}, Data: {args.data_set}, Seed: {args.seed}")
                    print("---------------------------------------")
                    
                else:
                    file_name = f"{args.mode}_{args.policy}_TB_Entropy_{args.Entropy}_{args.env}_dataset_{args.data_set}_{args.seed}"
                    print("---------------------------------------")
                    print(f"Mode: {args.mode}, Policy: {args.policy}_TB_Entropy_{args.Entropy}, Env: {args.env}, Data: {args.data_set}, Seed: {args.seed}")
                    print("---------------------------------------")    
        
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
        
        if args.save_model: 
            np.save(f"./results/{args.mode}/evaluation_{file_name}", evaluations)
            np.save(f"./results/{args.mode}/wallclock_time_time_{file_name}", wallclock_time)
            policy.save_actor(f"./Saved_models/{args.mode}/{file_name}/{file_name}")
            
    if args.mode == "off-on-RL":
        
        args.env = "MiniGrid-Empty-16x16-v0"
        
        file_name = f"{args.mode}_{args.policy}_{args.env}_{args.seed}"
        print("---------------------------------------")
        print(f"Mode: {args.mode}, Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")
        
        if not os.path.exists(f"./results/{args.mode}"):
            os.makedirs(f"./results/{args.mode}")
            
        if not os.path.exists(f"./Saved_models/{args.mode}/{file_name}"):
            os.makedirs(f"./Saved_models/{args.mode}/{file_name}")
        
        wallclock_time, evaluations, policy = train(args, args.seed)
        
        if args.save_model: 
            np.save(f"./results/{args.mode}/evaluation_{file_name}", evaluations)
            np.save(f"./results/{args.mode}/wallclock_time_{file_name}", wallclock_time)
            policy.save_actor(f"./Saved_models/{args.mode}/{file_name}/{file_name}")
            

        
        
    
   
                
                
                