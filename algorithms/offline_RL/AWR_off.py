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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AWR_off:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, Entropy = True,   
                 num_steps_per_rollout=2000, l_rate_actor=3e-4, l_rate_alpha=3e-4, discount=0.99, tau=0.005, beta=3, 
                 gae_gamma = 0.99, gae_lambda = 0.99, minibatch_size=64, num_epochs=10, alpha=0.2):
        
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
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        self.states = []
        self.actions = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        
        self.Entropy = Entropy
        self.target_entropy = -torch.FloatTensor([action_dim]).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr = l_rate_alpha) 
        self.alpha = alpha
        
        self.Total_t = 0
        self.Total_iter = 0
        
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
        
    def GAE(self, replay_buffer, ntrajs):
        states = []
        actions = []
        returns = []
        advantage = []
        gammas_list = []
        lambdas_list = []
        
        sampled_states, sampled_actions, sampled_rewards, sampled_lengths = replay_buffer.sample_trajectories(ntrajs)
        
        for l in range(ntrajs):
            traj_size = sampled_lengths[l]
            gammas = []
            lambdas = []
            for t in range(traj_size):
                gammas.append(self.gae_gamma**t)
                lambdas.append(self.gae_lambda**t)
                
            gammas_list.append(torch.FloatTensor(np.array(gammas)).to(device))
            lambdas_list.append(torch.FloatTensor(np.array(lambdas)).to(device))
            
        for l in range(ntrajs):
            
            with torch.no_grad():
                episode_states = sampled_states[l]
                episode_actions = sampled_actions[l]
                episode_rewards = sampled_rewards[l].squeeze() 
                episode_gammas = gammas_list[l]
                episode_lambdas = lambdas_list[l]
                
                traj_size = sampled_lengths[l] 
                
                self.actor.eval()
 
                episode_discounted_rewards = episode_gammas*episode_rewards
                episode_discounted_returns = torch.FloatTensor([episode_discounted_rewards[i:].sum() for i in range(traj_size)]).to(device)
                episode_returns = episode_discounted_returns
            
                current_values = self.actor.value_net(episode_states).detach()
                next_values = torch.cat((self.actor.value_net(episode_states)[1:], torch.FloatTensor([[0.]]).to(device))).detach()
                episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values
                episode_advantage = torch.FloatTensor([((episode_gammas*episode_lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(traj_size)]).to(device)
                
                states.append(episode_states)
                actions.append(episode_actions)
                returns.append(episode_returns)
                advantage.append(episode_advantage)
            
        return states, actions, returns, advantage
            
    def train(self, states, actions, returns, advantage):
        
        rollout_states = torch.cat(states)
        rollout_actions = torch.cat(actions)
        rollout_returns = torch.cat(returns)
        rollout_advantage = torch.cat(advantage)
        
        rollout_advantage = (rollout_advantage-rollout_advantage.mean())/(rollout_advantage.std()+1e-6)
        
        self.actor.train()
        
        self.num_steps_per_rollout = len(rollout_advantage)
        max_steps = self.num_epochs * (self.num_steps_per_rollout // self.minibatch_size)
        
        for _ in range(max_steps):
            
            minibatch_indices = np.random.choice(range(self.num_steps_per_rollout), self.minibatch_size, False)
            batch_states=rollout_states[minibatch_indices]
            batch_actions = rollout_actions[minibatch_indices]
            batch_returns = rollout_returns[minibatch_indices]
            batch_advantage = rollout_advantage[minibatch_indices]
                    
            if self.action_space == "Discrete":
                log_prob, log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)

            elif self.action_space == "Continuous": 
                log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)
                
            r = (log_prob_rollout).squeeze()
            weights = F.softmax(batch_advantage/self.beta, dim=0).detach()
            L_clip = r*weights
            L_vf = (self.actor.value_net(batch_states).squeeze() - batch_returns)**2
                
            self.actor_optimizer.zero_grad()
            
            if self.Entropy:
                _, log_pi_state = self.actor.sample(batch_states)
                loss = (-1) * (L_clip - L_vf - self.alpha*log_pi_state).mean()
            else:
                loss = (-1) * (L_clip - L_vf).mean()
            
            loss.backward()
            self.actor_optimizer.step()
            
            if self.Entropy: 

                alpha_loss = -(self.log_alpha * (log_pi_state + self.target_entropy).detach()).mean()
        
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
        
                self.alpha = self.log_alpha.exp()
        
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
        

        
        
        
        
        
        

            
            
        
            
            
            

        