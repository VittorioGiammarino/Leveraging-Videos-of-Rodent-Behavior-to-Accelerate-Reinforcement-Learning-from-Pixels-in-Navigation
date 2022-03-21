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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO_off:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, BC = False,
                 Entropy = False, GAE = False, num_steps_per_rollout=256, l_rate_actor=3e-4, gae_gamma = 0.99, 
                 gae_lambda = 0.99, epsilon = 0.3, c1 = 1, c2 = 1e-2, c3 = 1, minibatch_size=64, num_epochs=10):
        
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
        self.c3 = c3
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        self.Total_t = 0
        self.Total_iter = 0

        self.Entropy = Entropy
        self.GAE = GAE
        self.BC = BC
        
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
        
    def Calculate_Advantage(self, replay_buffer, ntrajs):

        returns = []
        advantage = []
        gammas_list = []
        lambdas_list = []
        
        sampled_states, sampled_actions, sampled_rewards, sampled_lengths = replay_buffer.sample_trajectories(ntrajs)
        
        if self.GAE:
            lambdas_list = []
            for l in range(ntrajs):
                gammas = []
                lambdas = []
                for t in range(sampled_lengths[l]):
                    gammas.append(self.gae_gamma**t)
                    lambdas.append(self.gae_lambda**t)
                    
                gammas_list.append(torch.FloatTensor(np.array(gammas)).to(device))
                lambdas_list.append(torch.FloatTensor(np.array(lambdas)).to(device))
                
        else:
            for l in range(ntrajs):
                gammas = []
                for t in range(sampled_lengths[l]):
                    gammas.append(self.gae_gamma**t)
                    
                gammas_list.append(torch.FloatTensor(np.array(gammas)).to(device)) 
        
        for l in range(ntrajs):
            
            with torch.no_grad():
                
                episode_states = sampled_states[l]
                episode_actions = sampled_actions[l]
                episode_rewards = sampled_rewards[l] 
                episode_gammas = gammas_list[l]
                
                K = sampled_lengths[l]
                    
                episode_discounted_rewards = episode_gammas*episode_rewards.squeeze() 
                episode_discounted_returns = torch.FloatTensor([sum(episode_discounted_rewards[i:]) for i in range(K)]).to(device)
                episode_returns = episode_discounted_returns/episode_gammas
                
                self.actor.eval()
                current_values = self.actor.value_net(episode_states).detach()
                next_values = torch.cat((self.actor.value_net(episode_states)[1:].detach(), torch.FloatTensor([[0.]]).to(device))).detach()
                episode_deltas = episode_rewards + self.gae_gamma*next_values - current_values
                
                if self.GAE:
                    episode_lambdas = lambdas_list[l]
                else:
                    _, log_prob_rollout = self.actor.sample_log(episode_states, episode_actions)
                    r = (torch.exp(log_prob_rollout)).squeeze()
                    try:
                        episode_lambdas = torch.FloatTensor([(r[:j]).prod() for j in range(K)]).to(device)
                    except:
                        episode_lambdas = r
                
                episode_advantage = torch.FloatTensor([((episode_gammas*episode_lambdas)[:K-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(K)]).to(device)
                
                returns.append(episode_returns)
                advantage.append(episode_advantage)
            
        return sampled_states, sampled_actions, returns, advantage
    
    def train(self, states, actions, returns, advantage):
        
        rollout_states = torch.cat(states)
        rollout_actions = torch.cat(actions)
        rollout_returns = torch.cat(returns)
        rollout_advantage = torch.cat(advantage)      
        
        rollout_advantage = (rollout_advantage-rollout_advantage.mean())/(rollout_advantage.std()+1e-6)
        
        self.actor.train()       
        max_steps = self.num_epochs * (self.num_steps_per_rollout // self.minibatch_size)
        
        for _ in range(max_steps):
            
            minibatch_indices = np.random.choice(range(self.num_steps_per_rollout), self.minibatch_size, False)
            batch_states = rollout_states[minibatch_indices]
            batch_actions = rollout_actions[minibatch_indices]
            batch_returns = rollout_returns[minibatch_indices]
            batch_advantage = rollout_advantage[minibatch_indices]     
            
            if self.action_space == "Discrete":
                log_prob, log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)

            elif self.action_space == "Continuous": 
                log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)
            
            if self.BC:
                action_prob = self.actor(batch_states)       
                m = F.one_hot(batch_actions.squeeze().cpu(), self.action_space_cardinality).float().to(device)
                recon_loss = F.mse_loss(action_prob, m)
            else:
                recon_loss = 0
            
            r = (log_prob_rollout).squeeze()
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
            loss = (-1) * (L_clip - self.c3*recon_loss - self.c1 * L_vf + self.c2 * S).mean()
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
        

        
        
        
        
        
        

            
            
        
            
            
            

        