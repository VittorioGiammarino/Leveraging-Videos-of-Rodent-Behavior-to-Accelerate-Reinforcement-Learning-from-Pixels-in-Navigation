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
from models.sample_models import Critic_flat

from models.on_off_obs_minigrid_models import SoftmaxActor
from models.on_off_obs_minigrid_models import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class on_off_AWAC_obs(object):
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, Entropy = False, 
                 Prioritized = False, l_rate_actor=1e-4, l_rate_critic=3e-4, l_rate_alpha=1e-4, 
                 discount=0.99, beta = 3, tau=0.005, alpha=0.2, critic_freq=2, l_rate_discr = 3e-8, intrinsic_reward = 0.001):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action        
        
        if np.isinf(action_space_cardinality):
            self.actor = TanhGaussianHierarchicalActor.NN_PI_LO(state_dim, action_dim, max_action).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Continuous"
            
        else:
            self.actor = SoftmaxActor(state_dim, action_space_cardinality).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Discrete"
            self.discriminator = Discriminator().to(device)
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=l_rate_discr)
                    
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.critic_freq = critic_freq
        
        self.Entropy = Entropy
        self.target_entropy = -torch.FloatTensor([action_dim]).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr = l_rate_alpha) 
        self.alpha = alpha
        
        self.Prioritized = Prioritized

        self.total_it = 0
        
        self.intrinsic_reward = 0.001
                
    def select_action(self, state):
        with torch.no_grad():
            if self.action_space == "Discrete":
                state = torch.FloatTensor(state.transpose(2,0,1)).unsqueeze(0).to(device)
                action, _ = self.actor.sample(state)
                return int((action).cpu().data.numpy().flatten())
            
            if self.action_space == "Continuous":
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action, _, _ = self.actor.sample_SAC_continuous(state)
                return (action).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer_online, off_policy_data, batch_size=128):
        self.total_it += 1

        # Sample replay buffer 
        if self.Prioritized:
            batch, weights, tree_idxs = replay_buffer_online.sample(batch_size)  
            state_on, action_on, next_state_on, reward_on, cost_on, not_done_on = batch
        else:
            state_on, action_on, next_state_on, reward_on, cost_on, not_done_on = replay_buffer_online.sample(batch_size)
        
        size_off_policy_data = len(off_policy_data)
        ind = np.random.randint(0, size_off_policy_data-1, size = batch_size)
        
        state_off = torch.FloatTensor(np.array([off_policy_data[ind]]).squeeze(0)).to(device)
        next_state_off = torch.FloatTensor(np.array([off_policy_data[ind+1]]).squeeze(0)).to(device)
        action_off = self.actor.sample_inverse_model(state_off, next_state_off).unsqueeze(1)
        reward_off_inv = self.actor.forward_inv_reward(state_off, next_state_off)
        reward_i = self.intrinsic_reward*torch.ones_like(reward_off_inv)  
        reward_off = (reward_off_inv + reward_i).to(device)
        
        state = torch.cat([state_on, state_off])
        action = torch.cat([action_on, action_off])
        next_state = torch.cat([next_state_on, next_state_off])
        reward = torch.cat([reward_on, reward_off])
        cost = torch.cat([cost_on, torch.zeros_like(cost_on, device=device)])
        not_done = torch.cat([not_done_on, torch.ones_like(not_done_on, device=device)])

        with torch.no_grad():
            # Select action according to policy and add clipped noise
			
            if self.action_space == "Discrete":
                next_action, log_pi_next_state = self.actor.sample(next_state)
                target_Q1, target_Q2 = self.actor.critic_target(next_state)
                current_target_Q1 = target_Q1.gather(1, next_action.detach().long().unsqueeze(-1)) 
                current_target_Q2 = target_Q2.gather(1, next_action.detach().long().unsqueeze(-1)) 
                
                if self.Entropy:
                    target_Q = (torch.min(current_target_Q1, current_target_Q2) - self.alpha*log_pi_next_state)
                    target_Q = reward-cost + not_done * self.discount * target_Q.sum(dim=1).unsqueeze(-1)
                else:
                    target_Q = torch.min(current_target_Q1, current_target_Q2)
                    target_Q = reward-cost + not_done * self.discount * target_Q.sum(dim=1).unsqueeze(-1)                  
                
            elif self.action_space == "Continuous":
                next_action, log_pi_next_state, _ = self.actor.sample_SAC_continuous(next_state)
                    
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                
                if self.Entropy:
                    target_Q = torch.min(target_Q1, target_Q2) - self.alpha*log_pi_next_state
                    target_Q = reward + not_done * self.discount * target_Q
                else:
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward + not_done * self.discount * target_Q

        if self.action_space == "Discrete":
            Q1, Q2 = self.actor.critic_net(state)
            current_Q1 = Q1.gather(1, action.detach().long()) 
            current_Q2 = Q2.gather(1, action.detach().long()) 
        
        elif self.action_space == "Continuous":
            #current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

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

        if self.action_space == "Discrete":
            Q1, Q2 = self.actor.critic_net(state)
            pi_action, log_pi_state = self.actor.sample(state)
            actor_Q1 = Q1.gather(1, pi_action.detach().long().unsqueeze(-1)) 
            actor_Q2 = Q2.gather(1, pi_action.detach().long().unsqueeze(-1)) 
            value_function = torch.min(actor_Q1, actor_Q2)
            
            old_Q1 = Q1.gather(1, action.detach().long()) 
            old_Q2 = Q2.gather(1, action.detach().long()) 
            old_Q = torch.min(old_Q1, old_Q2)
            adv_pi = old_Q - value_function
            weights = F.softmax(adv_pi/self.beta, dim=0).detach()
            _, log_pi_state_action = self.actor.sample_log(state, action)            
            
            if self.Entropy:
                actor_loss = (-1)*(-self.alpha*log_pi_state + log_pi_state_action*weights).mean()
            else:
                actor_loss = (-1)*(log_pi_state_action*weights).mean()
            
        elif self.action_space == "Continuous":
            pi_action, log_pi_state, _ = self.actor.sample_SAC_continuous(state)
            Q1, Q2 = self.critic(state, pi_action)
            value_function = torch.min(Q1,Q2)
            
            Q1_old, Q2_old = self.critic(state, action)
            Q_old_actions = torch.min(Q1_old, Q2_old)
            
            adv_pi = Q_old_actions - value_function
            weights = F.softmax(adv_pi/self.beta, dim=0).detach()
            log_pi_state_action = self.actor.sample_log(state, action)

            if self.Entropy:
                actor_loss = (-1)*(-self.alpha*log_pi_state + log_pi_state_action*weights).mean()
            else:
                actor_loss = (-1)*(log_pi_state_action*weights).mean()

        inverse_action_model_prob = self.actor.forward_inv_a(state_on, next_state_on)
        m = F.one_hot(action_on.squeeze().cpu(), self.action_space_cardinality).float().to(device)
        L_ia = F.mse_loss(inverse_action_model_prob, m)
        
        L_ir = F.mse_loss(reward_on, self.actor.forward_inv_reward(state_on, next_state_on))
                
        state_off_class = torch.zeros(batch_size, device=device)
        state_on_class = torch.ones(batch_size, device=device)
        criterion = torch.nn.BCELoss()
        
        d_loss_rollout = criterion(self.discriminator(self.actor.encode_image(state_on).detach()).squeeze(), state_on_class) 
        d_loss_obs = criterion(self.discriminator(self.actor.encode_image(state_off).detach()).squeeze(), state_off_class) 
        d_loss = 0.5*(d_loss_rollout + d_loss_obs)
        
        self.discriminator_optimizer.zero_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()
        
        encode_loss = criterion(self.discriminator(self.actor.encode_image(state_off)).squeeze(), state_on_class)   
			
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        loss = actor_loss + L_ia + L_ir + encode_loss
        loss.backward()
        self.actor_optimizer.step()
			        
        if self.Entropy: 

            alpha_loss = -(self.log_alpha * (log_pi_state + self.target_entropy).detach()).mean()
    
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
    
            self.alpha = self.log_alpha.exp()

        # Update the frozen target models
        if self.total_it % self.critic_freq == 0:
            for param, target_param in zip(self.actor.Q1.parameters(), self.actor.critic_target_Q1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.Q2.parameters(), self.actor.critic_target_Q2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
        
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))
        
    def save_critic(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
    
    def load_critic(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        
        
        
        
        
        

            
            
        
            
            
            

        