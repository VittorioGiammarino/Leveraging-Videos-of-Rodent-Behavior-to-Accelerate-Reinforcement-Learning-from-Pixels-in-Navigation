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


from models.sample_models import TanhGaussianHierarchicalActor
from models.sample_models import Value_net

from models.simple_minigird_models import SoftmaxHierarchicalActor
from models.simple_minigird_models import Value_net_CNN

from models.ResMLP import SoftmaxHierarchicalActorMLP
from models.ResMLP import ValueNetMLP

from Buffers.tb_buffer import GAE_TB_Buffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class A2C:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, Entropy = False,  
                 num_steps_per_rollout=2000, l_rate_actor=3e-4, gae_gamma = 0.99, gae_lambda = 0.99, 
                 epsilon = 0.2, c1 = 1, c2 = 1e-2, minibatch_size=64, num_epochs=10):
        
        if np.isinf(action_space_cardinality):
            self.actor = TanhGaussianHierarchicalActor.NN_PI_LO(state_dim, action_dim, max_action).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Continuous"
            
        else:
            self.actor = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_space_cardinality).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Discrete"
            
        # self.value_function = Value_net_CNN(state_dim).to(device)
        # self.value_function_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=l_rate_actor)
      
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action
        
        self.num_steps_per_rollout = num_steps_per_rollout
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        self.Total_t = 0
        self.Total_iter = 0

        self.Entropy = Entropy
        
        self.buffer = GAE_TB_Buffer()
        
    def reset_counters(self):
        self.Total_t = 0
        self.Total_iter = 0
        
    def select_action(self, state):
        with torch.no_grad():
            if self.action_space == "Discrete":
                state = torch.FloatTensor(state.transpose(2,0,1)).unsqueeze(0).to(device)
                action, _ = self.actor.sample(state)
                return int((action).cpu().data.numpy().flatten())
            
            if self.action_space == "Continuous":
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action, _, _ = self.actor.sample(state)
                return (action).cpu().data.numpy().flatten()
        
    def Generate_and_store_rollout(self, env, args):
        step = 0
        self.Total_iter += 1
        
        with torch.no_grad():
            while step < self.num_steps_per_rollout: 
                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_gammas = []
                episode_lambdas = []    
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
                        action = A2C.select_action(self, state)
                
                    episode_states.append(state.transpose(2,0,1))
                    episode_actions.append(action)
                    episode_gammas.append(self.gae_gamma**t)
                    episode_lambdas.append(self.gae_lambda**t)
                    
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
                episode_lambdas = torch.FloatTensor(np.array(episode_lambdas))
                    
                episode_discounted_rewards = episode_gammas*episode_rewards
                episode_discounted_returns = torch.FloatTensor([sum(episode_discounted_rewards[i:]) for i in range(t)])
                episode_returns = episode_discounted_returns/episode_gammas
                
                self.buffer.add(episode_states, episode_actions, episode_rewards, episode_returns, episode_gammas, episode_lambdas, t)
            
    def ADV_GAE(self, states, rewards, gammas, lambdas, episode_length):
        
        size = self.buffer.batch_size
        self.actor.eval()
        
        for l in range(size):
            with torch.no_grad():
                episode_states = states[l]
                episode_rewards = rewards[l]
                episode_gammas = gammas[l]
                episode_lambdas = lambdas[l]
                
                K = episode_length[l]
                
                current_values = self.actor.value_net(episode_states).detach()
                next_values = torch.cat((self.actor.value_net(episode_states)[1:], torch.FloatTensor([[0.]]).to(device))).detach()
                episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values
                episode_advantage = torch.FloatTensor([((episode_gammas*episode_lambdas)[:K-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(K)])
                
                self.buffer.add_Adv(l, episode_advantage)
    
    def train(self):
        
        states, actions, rewards, returns, gammas, lambdas, length = self.buffer.sample()
        self.ADV_GAE(states, rewards, gammas, lambdas, length)
        
        rollout_states = torch.cat(states)
        rollout_actions = torch.cat(actions)
        rollout_returns = torch.cat(returns)
        rollout_advantage = torch.cat(self.buffer.Buffer['advantage']) 
        
        rollout_advantage = (rollout_advantage-rollout_advantage.mean())/(rollout_advantage.std()+1e-6)
        
        self.actor.eval()
        with torch.no_grad():
            if self.action_space == "Discrete":
                _, old_log_prob_rollout = self.actor.sample_log(rollout_states, rollout_actions)
                old_log_prob_rollout = old_log_prob_rollout.detach()
            elif self.action_space == "Continuous": 
                old_log_prob_rollout = self.actor.sample_log(rollout_states, rollout_actions)
                old_log_prob_rollout = old_log_prob_rollout.detach()
        
        # self.value_function.train()
        self.actor.train()
        
        max_steps = self.num_epochs * (self.num_steps_per_rollout // self.minibatch_size)
        
        for _ in range(max_steps):
            
            minibatch_indices = np.random.choice(range(self.num_steps_per_rollout), self.minibatch_size, False)
            batch_states = rollout_states[minibatch_indices]
            batch_actions = rollout_actions[minibatch_indices]
            batch_returns = rollout_returns[minibatch_indices]
            batch_advantage = rollout_advantage[minibatch_indices].to(device)    
            
            if self.action_space == "Discrete":
                log_prob, log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)

            elif self.action_space == "Continuous": 
                log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)
                
            batch_old_log_pi = old_log_prob_rollout[minibatch_indices].to(device)  
                
            r = (torch.exp(log_prob_rollout - batch_old_log_pi)).squeeze()
            L_clip = r*batch_advantage
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
        

        
        
        
        
        
        

            
            
        
            
            
            

        