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
from torch.distributions.categorical import Categorical

# model for continuous action space environment
from models.simple_continuous_pixel_models import TanhGaussianActor
from models.on_off_obs_continuous_pixel_models import Encoder_Sawyer

from models.simple_minigird_models import SoftmaxHierarchicalActor
from models.on_off_obs_minigrid_models import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, Load_encoder = True,
                 Prioritized = False, l_rate_actor=3e-4, l_rate_critic=3e-4, discount=0.99, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        if np.isinf(action_space_cardinality):
            self.actor = TanhGaussianActor(state_dim, action_dim, max_action).to(device)
            self.encoder_on = Encoder_Sawyer(state_dim, action_dim).to(device)
            
            if Load_encoder:
                self.encoder_on.load_state_dict(torch.load("checkpoints/encoder.pt"))
            
              
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Continuous"
            
        else:
            self.actor = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_space_cardinality).to(device)
            self.encoder_on = Encoder(state_dim, action_space_cardinality).to(device)

            if Load_encoder:
                self.encoder_on.load_state_dict(torch.load("checkpoints/encoder.pt"))              
            
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Discrete"

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.Prioritized = Prioritized

        self.total_it = 0

    def select_action(self, state):
        with torch.no_grad():
            if self.action_space == "Discrete":
                state = torch.FloatTensor(state.transpose(2,0,1)).unsqueeze(0).to(device)
                embedding = self.encoder_on(state)
                action, _ = self.actor.sample(embedding)
                return int((action).cpu().data.numpy().flatten())
            
            if self.action_space == "Continuous":
                state = torch.FloatTensor(state.transpose(2,0,1)).unsqueeze(0).to(device)
                embedding = self.encoder_on(state)
                action, _, _ = self.actor.sample(embedding)
                return (action).cpu().data.numpy().flatten()
        
    def explore(self, state, expl_noise):
        with torch.no_grad():
            if self.action_space == "Discrete":
                state = torch.FloatTensor(state.transpose(2,0,1)).unsqueeze(0).to(device)
                embedding = self.encoder_on(state)
                prob_u = self.actor(embedding).cpu().data.numpy()
                noised_prob = prob_u + np.random.normal(0, expl_noise, size=self.action_space_cardinality)
                prob_u = np.exp(noised_prob) / np.sum(np.exp(noised_prob))
                prob_u = torch.FloatTensor(prob_u)
                m = Categorical(prob_u)
                action = m.sample()            
                return int(action.detach().data.numpy().flatten())
                
            if self.action_space == "Continuous":
                state = np.array(state)
                action = (TD3.select_action(self, state) + np.random.normal(0, self.max_action * expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action)
                return action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        if self.Prioritized:
            batch, weights, tree_idxs = replay_buffer.sample(batch_size)  
            state, action, next_state, reward, cost, not_done = batch
        else:
            state, action, next_state, reward, cost, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
			# Select action according to policy and add clipped noise
            state = self.encoder_on(state)
            next_state = self.encoder_on(next_state)
            
            if self.action_space == "Discrete":
                noise = (torch.randn_like(torch.zeros((batch_size, self.action_space_cardinality))) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(device)
                normalize = nn.Softmax(dim=1)
                next_action_prob = normalize(self.actor.actor_target_net(next_state) + noise)               
                distb = Categorical(next_action_prob)
                next_action = distb.sample()
                
                # Compute the target Q value
                target_Q1, target_Q2 = self.actor.critic_target(next_state)
                current_target_Q1 = target_Q1.gather(1, next_action.detach().long().unsqueeze(-1)) 
                current_target_Q2 = target_Q2.gather(1, next_action.detach().long().unsqueeze(-1)) 
                target_Q = (torch.min(current_target_Q1, current_target_Q2))
                target_Q = reward-cost + not_done * self.discount * target_Q.sum(dim=1).unsqueeze(-1)
                
            elif self.action_space == "Continuous":    
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor.actor_target_net(next_state) + noise).clamp(-self.max_action[0], self.max_action[0])
                # Compute the target Q value
                target_Q1, target_Q2 = self.actor.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

        if self.action_space == "Discrete":
            Q1, Q2 = self.actor.critic_net(state)
            current_Q1 = Q1.gather(1, action.detach().long()) 
            current_Q2 = Q2.gather(1, action.detach().long()) 
        
        elif self.action_space == "Continuous":
            #current Q estimates
            current_Q1, current_Q2 = self.actor.critic_net(state, action)

        if self.Prioritized:
            td_error_Q1 = torch.abs(current_Q1 - target_Q)
            td_error_Q2 = torch.abs(current_Q2 - target_Q)
            td_error = torch.min(td_error_Q1, td_error_Q2)
            critic_loss = torch.mean((current_Q1 - target_Q)**2 * weights) + torch.mean((current_Q2 - target_Q)**2 * weights)
        else:
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
        self.actor_optimizer.zero_grad()
        critic_loss.backward()
        self.actor_optimizer.step()
        
        if self.Prioritized:
            self.Buffer.update_priorities(tree_idxs, td_error.detach())

		# Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            self.actor.train()
            if self.action_space == "Discrete":
                Q1, Q2 = self.actor.critic_net(state)
                pi_action, log_pi_state = self.actor.sample(state)
                actor_Q1 = Q1.gather(1, pi_action.detach().long().unsqueeze(-1)) 
                actor_Q2 = Q2.gather(1, pi_action.detach().long().unsqueeze(-1)) 
                minQ = torch.min(actor_Q1, actor_Q2)
                actor_loss = (-1)*(minQ).mean()
            
            elif self.action_space == "Continuous":
    			# Compute actor loss
                pi_action = self.actor.deterministic_actor(state)
                Q1, Q2 = self.actor.critic_net(state, pi_action)
                minQ = torch.min(Q1, Q2)
                actor_loss = (-1)*(minQ).mean()
			
			# Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

			# Update the frozen target models
            for param, target_param in zip(self.actor.Q1.parameters(), self.actor.critic_target_Q1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.Q2.parameters(), self.actor.critic_target_Q2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.actor.parameters(), self.actor.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
        
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))
        self.actor_target = copy.deepcopy(self.actor)
        
    def save_critic(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
    
    def load_critic(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
		