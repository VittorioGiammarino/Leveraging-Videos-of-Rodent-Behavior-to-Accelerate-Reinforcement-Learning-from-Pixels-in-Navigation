#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:42:55 2021

@author: vittorio
"""

import numpy as np
import time

from evaluation import eval_policy


#############################################################################
# Offline RL
############################################################################

class run_TD3_BC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
		
            self.agent.train(replay_buffer)
            it_num+=1

            # Evaluate episode
            if it_num % 1 == 0:
                print(f"Iteration {it_num}")
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_SAC_BC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
		
            self.agent.train(replay_buffer)
            it_num+=1

            # Evaluate episode
            if it_num % 1 == 0:
                print(f"Iteration {it_num}")
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_AWAC_off:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
		
            self.agent.train(replay_buffer)
            it_num+=1

            # Evaluate episode
            if it_num % 1 == 0:
                print(f"Iteration {it_num}")
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_AWAC_Q_lambda_off_Peng:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
            
            states, actions, target_Q, advantage = self.agent.Q_lambda_Peng(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, target_Q, advantage)
            
            it_num+=1
            
            # Evaluate episode
            if it_num % 1 == 0:
                print(f"Iteration {it_num}")
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)   
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_AWAC_Q_lambda_off_Haru:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
            
            states, actions, target_Q, advantage = self.agent.Q_lambda_Haru(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, target_Q, advantage)
            
            it_num+=1
            
            # Evaluate episode
            if it_num % 1 == 0:
                print(f"Iteration {it_num}")
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)   
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_AWAC_Q_lambda_off_TB:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
            
            states, actions, target_Q, advantage = self.agent.TB_lambda(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, target_Q, advantage)
            
            it_num+=1
            
            # Evaluate episode
            if it_num % 1 == 0:
                print(f"Iteration {it_num}")
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)   
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_AWR_off:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
            
            states, actions, returns, advantage = self.agent.GAE(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, returns, advantage)
            
            it_num+=1
            
            # Evaluate episode
            if it_num % 1 == 0:
                print(f"Iteration {it_num}")
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)   
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_BC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
		
            self.agent.train(replay_buffer)
            it_num+=1

            # Evaluate episode
            if it_num % 1 == 0:
                print(f"Iteration {it_num}")
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent

################################################################################
# Interactive RL
################################################################################
            
class run_SAC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1
        start_time = time.time()

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
		
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                if args.action_space == "Continuous":
                    action = env.action_space.sample()
                elif args.action_space == "Discrete":
                    action = env.action_space.sample() 
            else:
                action = self.agent.select_action(state)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            try:
                done_bool = float(done) if episode_timesteps < env.env.unwrapped.max_steps else 0
            except:
                done_bool = float(done)

            replay_buffer.add(state, action, next_state, reward, 0, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                self.agent.train(replay_buffer)

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # Reset environment
                print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % (args.eval_freq*args.number_steps_per_iter) == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_AWAC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1
        start_time = time.time()

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
		
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                if args.action_space == "Continuous":
                    action = env.action_space.sample()
                elif args.action_space == "Discrete":
                    action = env.action_space.sample() 
            else:
                action = self.agent.select_action(state)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            try:
                done_bool = float(done) if episode_timesteps < env.env.unwrapped.max_steps else 0
            except:
                done_bool = float(done)

            replay_buffer.add(state, action, next_state, reward, 0, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                self.agent.train(replay_buffer)

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # Reset environment
                print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % (args.eval_freq*args.number_steps_per_iter) == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
        
class run_TD3:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1
        start_time = time.time()

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
		
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                if args.action_space == "Continuous":
                    action = env.action_space.sample() 
                elif args.action_space == "Discrete":
                    action = env.action_space.sample()  
            else:
                action = self.agent.explore(state, args.expl_noise)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            try:
                done_bool = float(done) if episode_timesteps < env.env.unwrapped.max_steps else 0
            except:
                done_bool = float(done)

            replay_buffer.add(state, action, next_state, reward, 0, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                self.agent.train(replay_buffer)

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # Reset environment
                print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % (args.eval_freq*args.number_steps_per_iter) == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent     

class run_AWAC_GAE:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.GAE(env, args)
            self.agent.train() 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
    
class run_AWAC_Q_lambda_Peng:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.Q_lambda_Peng(env, args)
            self.agent.train() 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
    
class run_AWAC_Q_lambda_Haru:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.Q_lambda_Haru(env, args)
            self.agent.train() 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
    
class run_AWAC_Q_lambda_TB:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.TB_lambda(env, args)
            self.agent.train() 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 

class run_PPO:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.GAE(env, args)
            self.agent.train() 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
    
class run_GePPO:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.Generate_and_store_rollout(env, args)
            self.agent.ADV_trace()
            self.agent.train() 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
    
#################################################################################
# on-off RL
#################################################################################

class run_on_off_SAC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer_online, replay_buffer_offline, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1
        start_time = time.time()

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
		
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                if args.action_space == "Continuous":
                    action = env.action_space.sample()
                elif args.action_space == "Discrete":
                    action = env.action_space.sample() 
            else:
                action = self.agent.select_action(state)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            try:
                done_bool = float(done) if episode_timesteps < env.env.unwrapped.max_steps else 0
            except:
                done_bool = float(done)

            replay_buffer_online.add(state, action, next_state, reward, 0, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                
                if args.mode == "on_off_RL_from_observations":
                    self.agent.train_inverse_models(replay_buffer_online)
                
                self.agent.train(replay_buffer_online, replay_buffer_offline)

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # Reset environment
                print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % (args.eval_freq*args.number_steps_per_iter) == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_on_off_AWAC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer_online, replay_buffer_offline, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
    
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1
        start_time = time.time()

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
		
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                if args.action_space == "Continuous":
                    action = env.action_space.sample()
                elif args.action_space == "Discrete":
                    action = env.action_space.sample() 
            else:
                action = self.agent.select_action(state)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            try:
                done_bool = float(done) if episode_timesteps < env.env.unwrapped.max_steps else 0
            except:
                done_bool = float(done)

            replay_buffer_online.add(state, action, next_state, reward, 0, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                
                if args.mode == "on_off_RL_from_observations":
                    self.agent.train_inverse_models(replay_buffer_online)
                
                self.agent.train(replay_buffer_online, replay_buffer_offline)

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # Reset environment
                print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % (args.eval_freq*args.number_steps_per_iter) == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_on_off_AWAC_Q_lambda_Peng:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.Q_lambda_Peng(env, args)
            
            if args.mode == "on_off_RL_from_observations":
                self.agent.train_inverse_models()
            
            states, actions, target_Q, advantage = self.agent.Q_lambda_Peng_off(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, target_Q, advantage)

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
    
class run_on_off_AWAC_Q_lambda_Haru:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.Q_lambda_Haru(env, args)
            
            if args.mode == "on_off_RL_from_observations":
                self.agent.train_inverse_models()
            
            states, actions, target_Q, advantage = self.agent.Q_lambda_Haru_off(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, target_Q, advantage)

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
    
class run_on_off_AWAC_TB:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.TB_lambda(env, args)
            
            if args.mode == "on_off_RL_from_observations":
                self.agent.train_inverse_models()
            
            states, actions, target_Q, advantage = self.agent.TB_lambda_off(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, target_Q, advantage)

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
    
class run_on_off_AWAC_GAE:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.GAE(env, args)
            
            if args.mode == "on_off_RL_from_observations":
                self.agent.train_inverse_models()
            
            states, actions, returns, advantage = self.agent.GAE_off(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, returns, advantage)

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
