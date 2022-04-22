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

from multiworld.envs.mujoco import register_goal_example_envs
from Buffers.vanilla_buffer import ReplayBuffer

import runner

from algorithms.RL.SAC import SAC
from algorithms.RL.AWAC import AWAC
from algorithms.RL.TD3 import TD3
from algorithms.RL.AWAC_GAE import AWAC_GAE
from algorithms.RL.AWAC_Q_lambda import AWAC_Q_lambda
from algorithms.RL.PPO import PPO
from algorithms.RL.GePPO import GePPO

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
        action_dim = 1

        action_space_cardinality = env.action_space.n
        max_action = np.nan
        min_action = np.nan
                
    state_dim = env.reset().shape
    
    #Buffers
    replay_buffer = ReplayBuffer(args.action_space, state_dim, action_dim)
    if args.mode == "on_off_RL_from_observations":
        
        replay_buffer_online = ReplayBuffer(args.action_space, state_dim, action_dim)
                    
        if args.data_set == 'human_expert_push':
            with open('data_set/human_data_set/obs_humans_processed.npy', 'rb') as f:
                off_policy_observations = np.load(f, allow_pickle=True)
                
            if args.reward_given:
                with open('data_set/human_data_set/reward_humans_processed.npy', 'rb') as f:
                    off_policy_rewards = np.load(f, allow_pickle=True)
                    
                off_policy_observations = [off_policy_observations, 10*off_policy_rewards]
                
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
             "intrinsic_reward": args.intrinsic_reward,
             "Load_encoder": args.load_encoder,
             "reward_given": args.reward_given
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
             "intrinsic_reward": args.intrinsic_reward,
             "Load_encoder": args.load_encoder,
             "reward_given": args.reward_given
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
             "number_obs_off_per_traj": args.number_obs_per_traj,
             "Load_encoder": args.load_encoder,
             "reward_given": args.reward_given
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
             "number_obs_off_per_traj": args.number_obs_per_traj,
             "Load_encoder": args.load_encoder,
             "reward_given": args.reward_given
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
             "number_obs_off_per_traj": args.number_obs_per_traj,
             "Load_encoder": args.load_encoder,
             "reward_given": args.reward_given
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
             "number_obs_off_per_traj": args.number_obs_per_traj,
             "Load_encoder": args.load_encoder,
             "reward_given": args.reward_given
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
             "min_action": min_action,
             "Load_encoder": args.load_encoder
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
             "Load_encoder": args.load_encoder
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
             "min_action": min_action,
             "Load_encoder": args.load_encoder
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
             "num_steps_per_rollout": args.number_steps_per_iter,
             "Load_encoder": args.load_encoder
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
             "num_steps_per_rollout": args.number_steps_per_iter,
             "Load_encoder": args.load_encoder,
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
             "num_steps_per_rollout": args.number_steps_per_iter,
             "Load_encoder": args.load_encoder,
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
             "num_steps_per_rollout": args.number_steps_per_iter,
             "Load_encoder": args.load_encoder,
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
             "num_steps_per_rollout": args.number_steps_per_iter,
             "Load_encoder": args.load_encoder,
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
             "Load_encoder": args.load_encoder
            }
    
            Agent_RL = GePPO(**kwargs)
            
            run_sim = runner.run_GePPO(Agent_RL)
            wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
            
            return wallclock_time, evaluation_RL, Agent_RL 
        
def train(args, seed): 
    
    register_goal_example_envs()
    env = gym.make(args.env)
          
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
    parser.add_argument("--mode", default="on_off_RL_from_observations", help='RL, on_off_RL_from_observations')     
    parser.add_argument("--env", default="Image48HumanLikeSawyerPushForwardEnvDiscrete-v1", help = 'Image48HumanLikeSawyerPushForwardEnv-v1, Image48HumanLikeSawyerPushForwardEnvDiscrete-v1')  
    parser.add_argument("--data_set", default="human_expert_push", help="human_expert_push")  
    parser.add_argument("--action_space", default="Discrete")  # Discrete or continuous
    
    parser.add_argument("--policy", default="SAC") 
    parser.add_argument("--seed", default=10, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--number_steps_per_iter", default=4096, type=int) # Number of steps between two evaluations (default Minigrid: 4096, default Sawyer: 20000)
    parser.add_argument("--eval_freq", default=1, type=int)          # How many iterations we evaluate
    parser.add_argument("--max_iter", default=100, type=int)    # Max number of iterations to run environment, max_steps = max_iter*number_steps_per_iter
    # RL
    parser.add_argument("--start_timesteps", default=5e3, type=int) # Time steps before training default=5e3 (default Minigrid: 5000, default Sawyer: 25000)
    parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise    
    # offline RL
    parser.add_argument("--ntrajs", default=10, type=int) #default: 10, number of off-policy trajectories 
    parser.add_argument("--number_obs_per_traj", default=100, type=int) # number of off-policy demonstrations or observations used for training at each iteration (default Minigrid: 100, default Sawyer: 500)
    parser.add_argument("--Entropy", action="store_true")
    parser.add_argument("--load_encoder", action="store_true")
    # from observations
    parser.add_argument("--domain_adaptation", action="store_true")
    parser.add_argument("--intrinsic_reward", default=0.1, type=float) #0.01 or 0.005 or 0
    parser.add_argument("--reward_given", action="store_true")
    # Evaluation
    parser.add_argument("--evaluation_episodes", default=10, type=int) #default: 10, number of episodes per evaluation
    parser.add_argument("--evaluation_max_n_steps", default = 1024, type=int) #default: 2000, max number of steps evaluation episode
    # Experiments
    parser.add_argument("--detect_gradient_anomaly", action="store_true")
    args = parser.parse_args()
    
    torch.autograd.set_detect_anomaly(args.detect_gradient_anomaly)
    
    assert args.env == 'Image48HumanLikeSawyerPushForwardEnv-v1' or args.env == 'Image48HumanLikeSawyerPushForwardEnvDiscrete-v1'
        
    if args.mode == "RL":
        
        if args.policy == "TD3" or args.policy == "SAC":
            file_name = f"{args.mode}_{args.policy}_loadEncoder_{args.load_encoder}_{args.env}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}, Load Encoder: {args.load_encoder}, Env: {args.env}, Seed: {args.seed}")
            print("---------------------------------------")
        else:
            file_name = f"{args.mode}_{args.policy}_Entropy_{args.Entropy}_loadEncoder_{args.load_encoder}_{args.env}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}_Entropy_{args.Entropy}, Load Encoder: {args.load_encoder}, Env: {args.env}, Seed: {args.seed}")
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
            file_name = f"{args.mode}_{args.policy}_{args.env}_dataset_{args.data_set}_loadEncoder_{args.load_encoder}_domain_adaptation_{args.domain_adaptation}_reward_given_{args.reward_given}_ri_{args.intrinsic_reward}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}, Load Encoder: {args.load_encoder}, Env: {args.env}, Data: {args.data_set}, Domain Adaptation: {args.domain_adaptation}, reward given: {args.reward_given}, ri: {args.intrinsic_reward}, Seed: {args.seed}")
            print("---------------------------------------")
        else:
            file_name = f"{args.mode}_{args.policy}_Entropy_{args.Entropy}_{args.env}_dataset_{args.data_set}_domain_adaptation_{args.domain_adaptation}_ri_{args.intrinsic_reward}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}_Entropy_{args.Entropy}, Load Encoder: {args.load_encoder}, Env: {args.env}, Data: {args.data_set}, Domain Adaptation: {args.domain_adaptation}, reward given: {args.reward_given}, ri: {args.intrinsic_reward}, Seed: {args.seed}")
            print("---------------------------------------")
        
        if not os.path.exists(f"./results/{args.mode}"):
            os.makedirs(f"./results/{args.mode}")
            
        if not os.path.exists(f"./Saved_models/{args.mode}/{file_name}"):
            os.makedirs(f"./Saved_models/{args.mode}/{file_name}")
        
        wallclock_time, evaluations, policy = train(args, args.seed)
        
        np.save(f"./results/{args.mode}/evaluation_{file_name}", evaluations)
        np.save(f"./results/{args.mode}/wallclock_time_{file_name}", wallclock_time)
        policy.save_actor(f"./Saved_models/{args.mode}/{file_name}/{file_name}")

        
        
    
   
                
                
                
