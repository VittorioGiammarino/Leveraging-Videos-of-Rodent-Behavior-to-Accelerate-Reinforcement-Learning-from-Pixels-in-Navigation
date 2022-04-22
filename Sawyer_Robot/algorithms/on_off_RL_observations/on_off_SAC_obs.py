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

from models.on_off_obs_continuous_pixel_models import TanhGaussianActor
from models.on_off_obs_continuous_pixel_models import Discriminator_GAN_Sawyer, Discriminator_WGAN_Sawyer
from models.on_off_obs_continuous_pixel_models import Encoder_Sawyer


from models.on_off_obs_minigrid_models import SoftmaxActor
from models.on_off_obs_minigrid_models import Discriminator_GAN, Discriminator_WGAN
from models.on_off_obs_minigrid_models import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class on_off_SAC_obs:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, Domain_adaptation = True, 
                 intrinsic_reward = 0.01, Load_encoder=True, reward_given = False, Prioritized = False, l_rate_actor=1e-4, 
                 l_rate_critic=3e-4, l_rate_alpha=1e-4, discount=0.99, tau=0.005, alpha=0.01, critic_freq=2, adversarial_loss = "wgan"): 
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action
        
        if np.isinf(action_space_cardinality):
            self.actor = TanhGaussianActor(state_dim, action_dim, max_action).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Continuous"
            
            # enconder on-policy data
            self.encoder_on = Encoder_Sawyer(state_dim, action_dim).to(device)
            
            if Load_encoder:
                self.encoder_on.load_state_dict(torch.load("checkpoints/encoder.pt"))
            
            if adversarial_loss == "gan":
                #encoder off-policy data
                self.encoder_off = copy.deepcopy(self.encoder_on)
                self.encoder_off_optimizer = torch.optim.Adam(self.encoder_off.parameters(), lr=2e-4, betas=(0.5, 0.999))
                
                # discriminator for domain adaptation
                self.discriminator = Discriminator_GAN_Sawyer().to(device)
                self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
                
            elif adversarial_loss == "wgan":
                #encoder off-policy data
                self.encoder_off = copy.deepcopy(self.encoder_on)
                self.encoder_off_optimizer = torch.optim.RMSprop(self.encoder_off.parameters(), lr=5e-5)
                
                # discriminator for domain adaptation
                self.discriminator = Discriminator_WGAN_Sawyer().to(device)
                self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=5e-5)
            
        else:
            self.action_space = "Discrete"
            self.actor = SoftmaxActor(state_dim, action_space_cardinality).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            
            # enconder on-policy data
            self.encoder_on = Encoder(state_dim, action_space_cardinality).to(device)
            
            if Load_encoder:
                self.encoder_on.load_state_dict(torch.load("checkpoints/encoder.pt"))
            
            if adversarial_loss == "gan":
                #encoder off-policy data
                self.encoder_off = copy.deepcopy(self.encoder_on)
                self.encoder_off_optimizer = torch.optim.Adam(self.encoder_off.parameters(), lr=2e-4, betas=(0.5, 0.999))
                
                # discriminator for domain adaptation
                self.discriminator = Discriminator_GAN().to(device)
                self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
                
            elif adversarial_loss == "wgan":
                #encoder off-policy data
                self.encoder_off = copy.deepcopy(self.encoder_on)
                self.encoder_off_optimizer = torch.optim.RMSprop(self.encoder_off.parameters(), lr=5e-5)
                
                # discriminator for domain adaptation
                self.discriminator = Discriminator_WGAN().to(device)
                self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=5e-5)
                    
        self.target_entropy = -torch.FloatTensor([action_dim]).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device = 'cuda')
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr = l_rate_alpha)     
        
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.critic_freq = critic_freq
        
        self.Prioritized = Prioritized

        self.total_it = 0
        
        self.intrinsic_reward = intrinsic_reward
        
        self.Domain_adaptation = Domain_adaptation
        self.adversarial_loss = adversarial_loss
        self.reward_given = reward_given
        
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
            
    def train_inverse_models(self, replay_buffer_online, batch_size=256):
        # Sample replay buffer 
        if self.Prioritized:
            batch, weights, tree_idxs = replay_buffer_online.sample(batch_size)  
            state_on, action_on, next_state_on, reward_on, cost_on, not_done_on = batch
        else:
            state_on, action_on, next_state_on, reward_on, cost_on, not_done_on = replay_buffer_online.sample(batch_size)
            
        embedding = self.encoder_on(state_on)
        embedding_next = self.encoder_on(next_state_on)
        
        if self.action_space == "Discrete":
            inverse_action_model_prob = self.actor.forward_inv_a(embedding, embedding_next)
            m = F.one_hot(action_on.squeeze().cpu(), self.action_space_cardinality).float().to(device)
            L_ia = F.mse_loss(inverse_action_model_prob, m)
            
        elif self.action_space == "Continuous":
            inverse_model_action = self.actor.sample_inverse_model(embedding, embedding_next)
            L_ia = F.mse_loss(inverse_model_action, action_on)
        
        L_ir = F.mse_loss(reward_on, self.actor.forward_inv_reward(embedding, embedding_next))
        
        self.actor_optimizer.zero_grad()
        loss = L_ia + L_ir 
        loss.backward()
        self.actor_optimizer.step()
            
    def train(self, replay_buffer_online, off_policy_data_tot, batch_size=128):
        self.total_it += 1

        # Sample replay buffer 
        if self.Prioritized:
            batch, weights, tree_idxs = replay_buffer_online.sample(batch_size)  
            state_on, action_on, next_state_on, reward_on, cost_on, not_done_on = batch
        else:
            state_on, action_on, next_state_on, reward_on, cost_on, not_done_on = replay_buffer_online.sample(batch_size)
        
        if self.reward_given:
            off_policy_reward = off_policy_data_tot[1]
            off_policy_data = off_policy_data_tot[0]
            reward_given = True
        else:
            reward_given = False
            off_policy_data = off_policy_data_tot
        
        size_off_policy_data = len(off_policy_data)
        ind = np.random.randint(0, size_off_policy_data-1, size = batch_size)
        
        state_off = torch.FloatTensor(np.array([off_policy_data[ind]]).squeeze(0)).to(device)
        next_state_off = torch.FloatTensor(np.array([off_policy_data[ind+1]]).squeeze(0)).to(device)
        
        with torch.no_grad():
            embedding_off = self.encoder_off(state_off)
            next_embedding_off = self.encoder_off(next_state_off)    
            
            if self.action_space == "Discrete":
                action_off = self.actor.sample_inverse_model(embedding_off, next_embedding_off).unsqueeze(1)
                
            elif self.action_space == "Continuous":
                action_off = self.actor.sample_inverse_model(embedding_off, next_embedding_off)
                
            if reward_given:
                reward_off_inv = torch.FloatTensor(off_policy_reward[ind]).to(device)
                reward_i = self.intrinsic_reward*torch.ones_like(reward_off_inv)     
                reward_off = (reward_off_inv + reward_i).to(device)
            else:
                reward_off_inv = self.actor.forward_inv_reward(embedding_off, next_embedding_off)
                reward_i = self.intrinsic_reward*torch.ones_like(reward_off_inv)  
                reward_off = (reward_off_inv + reward_i).to(device)
            
            embedding_on = self.encoder_on(state_on)
            next_embedding_on = self.encoder_on(next_state_on)
        
        embedding = torch.cat([embedding_on, embedding_off])
        action = torch.cat([action_on, action_off])
        next_embedding = torch.cat([next_embedding_on, next_embedding_off])
        reward = torch.cat([reward_on, reward_off])
        cost = torch.cat([cost_on, torch.zeros_like(cost_on, device=device)])
        not_done = torch.cat([not_done_on, torch.ones_like(not_done_on, device=device)])

        with torch.no_grad():
            # Select action according to policy and add clipped noise
			
            if self.action_space == "Discrete":
                # Compute the target Q value
                next_action, log_pi_next_state = self.actor.sample(next_embedding)
                target_Q1, target_Q2 = self.actor.critic_target(next_embedding)
                current_target_Q1 = target_Q1.gather(1, next_action.detach().long().unsqueeze(-1)) 
                current_target_Q2 = target_Q2.gather(1, next_action.detach().long().unsqueeze(-1)) 
                target_Q = (torch.min(current_target_Q1, current_target_Q2) - self.alpha*log_pi_next_state)
                target_Q = reward-cost + not_done * self.discount * target_Q.sum(dim=1).unsqueeze(-1)
                
            elif self.action_space == "Continuous":
                next_action, log_pi_next_state, _ = self.actor.sample(next_embedding)
                    
                # Compute the target Q value
                target_Q1, target_Q2 = self.actor.critic_target(next_embedding, next_action)
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha*log_pi_next_state
                target_Q = reward-cost + not_done * self.discount * target_Q

        if self.action_space == "Discrete":
            Q1, Q2 = self.actor.critic_net(embedding)
            current_Q1 = Q1.gather(1, action.detach().long()) 
            current_Q2 = Q2.gather(1, action.detach().long()) 
        
        elif self.action_space == "Continuous":
            #current Q estimates
            current_Q1, current_Q2 = self.actor.critic_net(embedding, action)

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
            Q1, Q2 = self.actor.critic_net(embedding)
            pi_action, log_pi_state = self.actor.sample(embedding)
            actor_Q1 = Q1.gather(1, pi_action.detach().long().unsqueeze(-1)) 
            actor_Q2 = Q2.gather(1, pi_action.detach().long().unsqueeze(-1)) 
            minQ = torch.min(actor_Q1, actor_Q2)
      
            actor_loss = (self.alpha*log_pi_state-minQ).mean()
            
        elif self.action_space == "Continuous":
            pi_action, log_pi_state, _ = self.actor.sample(embedding)
            Q1, Q2 = self.actor.critic_net(embedding, pi_action)
            minQ = torch.min(Q1,Q2)

            actor_loss = ((self.alpha*log_pi_state)-minQ).mean()
            
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.Domain_adaptation:
            
            if self.adversarial_loss == 'gan':
                
                obs_class = torch.zeros(batch_size, device=device)
                rollout_class = torch.ones(batch_size, device=device)
                criterion = torch.nn.BCELoss()
                
                # -----------------
                #  Train Encoder off
                # -----------------
                
                self.encoder_off_optimizer.zero_grad()
                
                embedding_off = self.encoder_off(state_off)
                encode_loss = criterion(self.discriminator(embedding_off).squeeze(), rollout_class)

                encode_loss.backward()
                self.encoder_off_optimizer.step()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                self.discriminator_optimizer.zero_grad()
                
                real_embedding = self.encoder_on(state_on).detach()
                d_loss_rollout = criterion(self.discriminator(real_embedding).squeeze(), rollout_class) 
                d_loss_obs = criterion(self.discriminator(embedding_off.detach()).squeeze(), obs_class) 
                d_loss = 0.5*(d_loss_rollout + d_loss_obs)
                
                d_loss.backward()
                self.discriminator_optimizer.step()
                
            elif self.adversarial_loss == 'wgan':
                
                clip_value = 0.01
                n_critic = 5
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                self.discriminator_optimizer.zero_grad()
                
                embedding_off = self.encoder_off(state_on).detach().squeeze()
                real_embedding = self.encoder_on(state_off).detach().squeeze()
                
                d_loss = -torch.mean(self.discriminator(real_embedding)) + torch.mean(self.discriminator(embedding_off))
                
                d_loss.backward()
                self.discriminator_optimizer.step()
                
                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)
                    
                # Train the encoder off every n_critic iterations
                if self.total_it % n_critic == 0:
                    
                    # -----------------
                    #  Train Encoder off
                    # -----------------
                    
                    self.encoder_off_optimizer.zero_grad()
                    
                    embedding_off = self.encoder_off(state_off).squeeze()
                    encode_loss = -torch.mean(self.discriminator(embedding_off))
    
                    encode_loss.backward()
                    self.encoder_off_optimizer.step()        
        
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
		
        
        
        
        
        

            
            
        
            
            
            

        