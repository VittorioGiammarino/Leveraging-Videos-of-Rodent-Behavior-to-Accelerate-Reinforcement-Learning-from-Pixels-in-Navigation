#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:54:01 2021

@author: vittorio
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# model for continuous action space environment
from models.simple_continuous_pixel_models import TanhGaussianActor

from models.simple_minigird_models import SoftmaxHierarchicalActor

from Buffers.v_trace_buffer import V_trace_Buffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GePPO:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, Entropy = False,
                 num_steps_per_rollout=5000, l_rate_actor=1e-4, gae_gamma = 0.99,  
                 epsilon = 0.1, c1 = 1, c2 = 1e-2, minibatch_size=64, num_epochs=5, N_old_policies = 4, c_trunc = 1):
        
        if np.isinf(action_space_cardinality):
            self.actor = TanhGaussianActor(state_dim, action_dim, max_action)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Continuous"
            
        else:
            self.actor = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_space_cardinality)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Discrete"
                  
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action
        
        self.num_steps_per_rollout = num_steps_per_rollout
        self.gae_gamma = gae_gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        self.N_old_policies = N_old_policies
        self.c_trunc = c_trunc
        self.buffer = V_trace_Buffer(N_old_policies)
        
        self.Total_t = 0
        self.Total_iter = 0
        
        self.Entropy = Entropy
        
    def reset_counters(self):
        self.Total_t = 0
        self.Total_iter = 0
        
    def select_action(self, state):
        with torch.no_grad():
            if self.action_space == "Discrete":
                state = torch.FloatTensor(state.transpose(2,0,1)).unsqueeze(0)
                action, _ = self.actor.sample(state)
                return int((action).cpu().data.numpy().flatten())
            
            if self.action_space == "Continuous":
                state = torch.FloatTensor(state.transpose(2,0,1)).unsqueeze(0)
                action, _, _ = self.actor.sample(state)
                return (action).cpu().data.numpy().flatten()
        
    def Generate_and_store_rollout(self, env, args):
        step = 0
        self.Total_iter += 1
        
        self.buffer.clear()
        
        with torch.no_grad():
            while step < self.num_steps_per_rollout: 
                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_gammas = []
                episode_rewards = []
                episode_gammas = []
                state, done = env.reset(), False
                t=0
                episode_reward = 0
    
                while not done and step < self.num_steps_per_rollout:    
                 # Select action randomly or according to policy
                    if self.Total_t < args.start_timesteps:
                       if args.action_space == "Continuous":
                           action = env.action_space.sample() 
                       elif args.action_space == "Discrete":
                           action = env.action_space.sample()  
                    else:
                        action = GePPO.select_action(self, state)
                    
                    episode_states.append(state.transpose(2,0,1))
                    episode_actions.append(action)
                    episode_gammas.append(self.gae_gamma**t)
                    
                    state, reward, done, _ = env.step(action)
                    
                    episode_rewards.append(reward)
                
                    t+=1
                    step+=1
                    episode_reward+=reward
                    self.Total_t += 1
                            
                if done: 
                    # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                    print(f"Total T: {self.Total_t}, Iter Num: {self.Total_iter}, Episode T: {t} Reward: {episode_reward:.3f}")
                    
                episode_states = torch.FloatTensor(np.array(episode_states))
                
                if self.action_space == "Discrete":
                    episode_actions = torch.LongTensor(np.array(episode_actions))
                elif self.action_space == "Continuous":
                    episode_actions = torch.FloatTensor(np.array(episode_actions))
                    
                episode_rewards = torch.FloatTensor(np.array(episode_rewards))
                episode_gammas = torch.FloatTensor(np.array(episode_gammas))      
                    
                episode_discounted_rewards = episode_gammas*episode_rewards
                episode_discounted_returns = torch.FloatTensor([sum(episode_discounted_rewards[i:]) for i in range(t)])
                episode_returns = episode_discounted_returns/episode_gammas
                    
                if self.action_space == "Discrete":
                    _, old_log_prob_rollout = self.actor.sample_log(episode_states, episode_actions)
                    old_log_prob_rollout = old_log_prob_rollout.data.cpu().detach()
                elif self.action_space == "Continuous": 
                    old_log_prob_rollout = self.actor.sample_log(episode_states, episode_actions)
                    old_log_prob_rollout = old_log_prob_rollout.detach().data.cpu().detach()
                     
                self.buffer.add(episode_states, episode_actions, episode_rewards, episode_returns, episode_gammas, old_log_prob_rollout, t)
            
        self.buffer.update_counters()
    
    def ADV_trace(self):
        
        stored_policies = self.buffer.size
        
        with torch.no_grad():
        
            for i in range(stored_policies):
                
                states = self.buffer.Buffer[i]['states']
                actions = self.buffer.Buffer[i]['actions']
                rewards = self.buffer.Buffer[i]['rewards']
                gammas = self.buffer.Buffer[i]['gammas']
                log_pi_old = self.buffer.Buffer[i]['log_pi_old']
                episode_length = self.buffer.Buffer[i]['episode_length']
                
                self.buffer.clear_Adv(i)
                
                for l in range(len(episode_length)):     
                        
                    episode_states = states[l]
                    episode_actions = actions[l]
                    episode_rewards = rewards[l]
                    episode_gammas = gammas[l]
                    episode_log_pi_old = log_pi_old[l]
                    
                    K = episode_length[l]
                                    
                    self.actor.eval()
                    current_values = self.actor.value_net(episode_states).detach().cpu()
                    next_values = torch.cat((self.actor.value_net(episode_states)[1:].cpu(), torch.FloatTensor([[0.]]))).detach()
                    episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values  
                    
                    if self.action_space == "Discrete":
                        _, log_prob_rollout_full = self.actor.sample_log(episode_states, episode_actions)
                        log_prob_rollout_full = log_prob_rollout_full.data.cpu().detach()
                    elif self.action_space == "Continuous": 
                        log_prob_rollout_full = self.actor.sample_log(episode_states, episode_actions)
                        log_prob_rollout_full = log_prob_rollout_full.detach()
                    
                    episode_adv = []
                    
                    for j in range(K):
                        
                        try:
                            log_prob_rollout = log_prob_rollout_full[j:]                                
                            r = (torch.exp(log_prob_rollout - episode_log_pi_old[j:])).squeeze()
                            r_trunc = torch.min(self.c_trunc*torch.ones_like(r), r)
                            pi_adjust = torch.FloatTensor([(r_trunc[:k]).prod() for k in range(1,len(r_trunc))])
                            pi_adjust_full = torch.cat((torch.FloatTensor([1.]), pi_adjust))
                            
                        except:
                            pi_adjust_full = torch.FloatTensor([1.])
                        
                        episode_adv.append((episode_gammas[:K-j].unsqueeze(-1)*pi_adjust_full*episode_deltas[j:]).sum())
                        
                    episode_advantage = torch.FloatTensor(episode_adv)
                    self.buffer.add_Adv(i, episode_advantage.detach())
    
    def train(self):
        
        stored_policies = self.buffer.size
        max_steps = self.num_epochs * (self.num_steps_per_rollout // self.minibatch_size) * stored_policies
        
        current_policy_log_prob = []
        for k in range(stored_policies):
            
            with torch.no_grad():
                rollout_states = torch.cat(self.buffer.Buffer[k]['states'])
                rollout_actions = torch.cat(self.buffer.Buffer[k]['actions'])
                
                if self.action_space == "Discrete":
                    log_prob, log_prob_rollout = self.actor.sample_log(rollout_states, rollout_actions)
    
                elif self.action_space == "Continuous": 
                    log_prob_rollout = self.actor.sample_log(rollout_states, rollout_actions)
                    
                current_policy_log_prob.append(log_prob_rollout.detach())
        
        self.actor.train().to(device)
        for s in range(max_steps):
            
            policy_number = np.random.randint(0, stored_policies)
            rollout_states = torch.cat(self.buffer.Buffer[policy_number]['states'])
            
            if self.action_space == "Discrete":
                rollout_actions = torch.cat(self.buffer.Buffer[policy_number]['actions'])
            elif self.action_space == "Continuous":
                rollout_actions = torch.cat(self.buffer.Buffer[policy_number]['actions'])
            
            rollout_returns = torch.cat(self.buffer.Buffer[policy_number]['returns'])
            rollout_advantage = torch.cat(self.buffer.Buffer[policy_number]['advantage'])
            rollout_advantage = (rollout_advantage-rollout_advantage.mean())/(rollout_advantage.std()+1e-6)
            
            old_log_prob_rollout = torch.cat(self.buffer.Buffer[policy_number]['log_pi_old']) 
            current_log_prob_rollout = current_policy_log_prob[policy_number]
        
            minibatch_indices = np.random.choice(range(self.num_steps_per_rollout), self.minibatch_size, False)
            batch_states = rollout_states[minibatch_indices].to(device)
            batch_actions = rollout_actions[minibatch_indices].to(device)
            batch_returns = rollout_returns[minibatch_indices].to(device)
            batch_advantage = rollout_advantage[minibatch_indices].to(device)
            
            if self.action_space == "Discrete":
                log_prob, log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)

            elif self.action_space == "Continuous": 
                log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)
                
            batch_old_log_pi = old_log_prob_rollout[minibatch_indices].to(device)
            batch_current_policy_log_pi = current_log_prob_rollout[minibatch_indices].to(device)
                
            r = (torch.exp(log_prob_rollout - batch_old_log_pi)).squeeze()
            r_bound = (torch.exp(batch_current_policy_log_pi - batch_old_log_pi)).squeeze()
            clipping = self.epsilon*torch.ones_like(r_bound)
            clipped_r = torch.where(r > r_bound+clipping, r_bound+clipping, r)
            clipped_r = torch.where(r < r_bound-clipping, r_bound-clipping, r)
            L_clip = torch.minimum(r*batch_advantage, clipped_r*batch_advantage)
            L_vf = (self.actor.value_net(batch_states).squeeze() - batch_returns)**2
            
            if self.action_space == "Discrete":
                if self.Entropy:
                    S = (-1)*torch.sum(torch.exp(log_prob)*log_prob, 1)
                else:
                    S = torch.zeros_like(torch.sum(torch.exp(log_prob)*log_prob, 1))
                    
            elif self.action_space == "Continuous": 
                if self.Entropy:
                    S = self.actor.Distb(batch_states).entropy()
                else:
                    S = torch.zeros_like(self.actor.Distb(batch_states).entropy())
                
            self.actor_optimizer.zero_grad()
            loss = (-1) * (L_clip - self.c1 * L_vf + self.c2 * S).mean()
            loss.backward()
            self.actor_optimizer.step()      
            
        self.actor.cpu()
        
    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
    
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))

    def save_critic(self, filename):
        torch.save(self.value_function.state_dict(), filename + "_value_function")
        torch.save(self.value_function_optimizer.state_dict(), filename + "_value_function_optimizer")
    
    def load_critic(self, filename):
        self.value_function.load_state_dict(torch.load(filename + "_value_function"))      
        self.value_function_optimizer.load_state_dict(torch.load(filename + "_value_function_optimizer")) 
        

        
        
        
        
        
        

            
            
        
            
            
            

        