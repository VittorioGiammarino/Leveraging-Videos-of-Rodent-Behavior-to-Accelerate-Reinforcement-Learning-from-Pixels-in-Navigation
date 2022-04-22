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

from models.RL_from_videos_model import SoftmaxActor
from models.RL_from_videos_model import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO_from_videos:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, external_data_set, number_obs_per_iter,
                 num_steps_per_rollout=2000, l_rate_actor=3e-4, l_rate_discr = 3e-8, gae_gamma = 0.99, gae_lambda = 0.99, 
                 epsilon = 0.3, c1 = 1, c2 = 1e-2, minibatch_size=64, num_epochs=10):
        
        if np.isinf(action_space_cardinality):
            self.actor = TanhGaussianHierarchicalActor.NN_PI_LO(state_dim, action_dim, max_action).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Continuous"
            
        else:
            self.actor = SoftmaxActor(state_dim, action_space_cardinality)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Discrete"
            self.discriminator = Discriminator().to(device)
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=l_rate_discr)
            
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
        self.states = []
        self.actions = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        
        self.off_policy_data = external_data_set
        self.number_obs_per_iter = number_obs_per_iter
        
    def reset_counters(self):
        self.Total_t = 0
        self.Total_iter = 0
        
    def select_action(self, state):
        if self.action_space == "Discrete":
            state = torch.FloatTensor(state.transpose(2,0,1)).unsqueeze(0)
            action, _ = self.actor.sample(state)
            return int((action).cpu().data.numpy().flatten())
        
        if self.action_space == "Continuous":
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action, _, _ = self.actor.sample(state)
            return (action).cpu().data.numpy().flatten()
        
    def GAE(self, env, args):
        step = 0
        self.Total_iter += 1
        self.states = []
        self.actions = []
        self.rewards = []
        self.returns = []
        self.advantage = []
        self.gammas = []
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
                    action = PPO_from_videos.select_action(self, state)
            
                self.states.append(state.transpose(2,0,1))
                self.actions.append(action)
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
            episode_actions = torch.FloatTensor(np.array(episode_actions))
            episode_rewards = torch.FloatTensor(np.array(episode_rewards))
            episode_gammas = torch.FloatTensor(np.array(episode_gammas))
            episode_lambdas = torch.FloatTensor(np.array(episode_lambdas))
                
            episode_discounted_rewards = episode_gammas*episode_rewards
            episode_discounted_returns = torch.FloatTensor([sum(episode_discounted_rewards[i:]) for i in range(t)])
            episode_returns = episode_discounted_returns/episode_gammas
            
            self.returns.append(episode_returns)
            self.actor.eval()
            current_values = self.actor.value_net(episode_states).detach().cpu()
            next_values = torch.cat((self.actor.value_net(episode_states)[1:].cpu(), torch.FloatTensor([[0.]]))).detach()
            episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values
            episode_advantage = torch.FloatTensor([((episode_gammas*episode_lambdas)[:t-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(t)])
            
            self.rewards.append(episode_rewards)
            self.advantage.append(episode_advantage)
            self.gammas.append(episode_gammas)
            
        rollout_states = torch.FloatTensor(np.array(self.states))
        rollout_actions = torch.FloatTensor(np.array(self.actions))

        return rollout_states, rollout_actions
    
    def TB_one(self, intrinsic_reward = 0.001):
        
        size_data_set = len(self.off_policy_data)
        ind = np.random.randint(0, size_data_set-self.number_obs_per_iter-1, size=1)
        
        states = torch.FloatTensor(np.array([self.off_policy_data[ind+i] for i in range(self.number_obs_per_iter)]).squeeze(1))
        next_states = torch.FloatTensor(np.array([self.off_policy_data[ind+i+1] for i in range(self.number_obs_per_iter)]).squeeze(1))
        actions = self.actor.sample_inverse_model(states, next_states)
        rewards = self.actor.forward_inv_reward(states, next_states)
        rewards_i = intrinsic_reward*torch.ones_like(rewards)    
        rewards = (rewards+rewards_i).squeeze()
        
        gammas = []
        
        K = self.number_obs_per_iter
        
        for t in range(K):
            gammas.append(self.gae_gamma**t)
            
        gammas = torch.FloatTensor(np.array(gammas))
        
        discounted_rewards = gammas*rewards
        discounted_returns = torch.FloatTensor([sum(discounted_rewards[i:]) for i in range(K)])
        returns = discounted_returns/gammas
            
        self.actor.eval()
        current_values = self.actor.value_net(states).detach().cpu()
        next_values = torch.cat((self.actor.value_net(states)[1:].cpu(), torch.FloatTensor([[0.]]))).detach()
        episode_deltas = rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values  
            
        _, log_prob_rollout = self.actor.sample_log(states, actions)
        r = (torch.exp(log_prob_rollout)).squeeze()
        
        try:
            episode_lambdas = torch.FloatTensor([(r[:j]).prod() for j in range(K)])
        except:
            episode_lambdas = r
            
        advantage = torch.FloatTensor([((gammas*(episode_lambdas))[:K-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(K)])
        
        return states, actions, returns, advantage, gammas
    
    def train(self, Entropy = False):
        
        rollout_states = torch.FloatTensor(np.array(self.states))
        
        if self.action_space == "Discrete":
            rollout_actions = torch.LongTensor(np.array(self.actions))
        elif self.action_space == "Continuous":
            rollout_actions = torch.FloatTensor(np.array(self.actions))
        
        rollout_returns = torch.cat(self.returns)
        rollout_advantage = torch.cat(self.advantage)
        rollout_gammas = torch.cat(self.gammas)  
        rollout_rewards = torch.cat(self.rewards)
        
        off_policy_states, off_policy_actions, off_policy_returns, off_policy_advantage, off_policy_gammas = self.TB_one()
        
        
        states = torch.cat((rollout_states, off_policy_states))
        actions = torch.cat((rollout_actions, off_policy_actions))
        returns = torch.cat((rollout_returns, off_policy_returns))
        advantage = torch.cat((rollout_advantage, off_policy_advantage))
        
        advantage = (advantage-advantage.mean())/(advantage.std()+1e-6)
        
        self.actor.eval()
        
        if self.action_space == "Discrete":
            _, old_log_prob_rollout = self.actor.sample_log(states, actions)
            old_log_prob_rollout = old_log_prob_rollout.detach()
        elif self.action_space == "Continuous": 
            old_log_prob_rollout = self.actor.sample_log(states, actions)
            old_log_prob_rollout = old_log_prob_rollout.detach()
        
        # self.value_function.train()
        self.actor.train().to(device)
        
        tot_samples = self.num_steps_per_rollout + self.number_obs_per_iter
        max_steps = self.num_epochs * (tot_samples // self.minibatch_size)
        
        for _ in range(max_steps):
            
            minibatch_indices = np.random.choice(range(tot_samples), self.minibatch_size, False)
            batch_states = states[minibatch_indices].to(device)
            batch_actions = actions[minibatch_indices].to(device)
            batch_returns = returns[minibatch_indices].to(device)
            batch_advantage = advantage[minibatch_indices].to(device)   
            
            if self.action_space == "Discrete":
                log_prob, log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)

            elif self.action_space == "Continuous": 
                log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)
                
            batch_old_log_pi = old_log_prob_rollout[minibatch_indices].to(device)  
                
            r = (torch.exp(log_prob_rollout - batch_old_log_pi)).squeeze()
            L_clip = torch.minimum(r*batch_advantage, torch.clip(r, 1-self.epsilon, 1+self.epsilon)*batch_advantage)
            L_vf = (self.actor.value_net(batch_states).squeeze() - batch_returns)**2
            
            if self.action_space == "Discrete":
                if Entropy:
                    S = (-1)*torch.sum(torch.exp(log_prob)*log_prob, 1)
                else:
                    S = torch.zeros_like(torch.sum(torch.exp(log_prob)*log_prob, 1))
                    
            elif self.action_space == "Continuous": 
                if Entropy:
                    S = self.actor.Distb(batch_states).entropy()
                else:
                    S = torch.zeros_like(self.actor.Distb(batch_states).entropy())
                    
            minibatch_indices_ims = np.random.choice(range(self.num_steps_per_rollout-1), self.minibatch_size, False)
            states_ims = rollout_states[minibatch_indices_ims].to(device)
            next_states_ims = rollout_states[minibatch_indices_ims + 1].to(device)
            rewards = rollout_rewards[minibatch_indices_ims].to(device)
            L_ia = (batch_actions.squeeze() - self.actor.sample_inverse_model(states_ims, next_states_ims))**2
            L_ir = (rewards.squeeze() - self.actor.forward_inv_reward(states_ims, next_states_ims).squeeze())**2
            
            minibatch_indices_obs = np.random.choice(range(self.number_obs_per_iter), self.minibatch_size, False)
            state_obs = off_policy_states[minibatch_indices_obs].to(device)
            
            obs_class = torch.zeros(self.minibatch_size, device=device)
            rollout_class = torch.ones(self.minibatch_size, device=device)
            criterion = torch.nn.BCELoss()
            
            d_loss_rollout = criterion(self.discriminator(self.actor.encode_image(states_ims).detach()).squeeze(), rollout_class) 
            d_loss_obs = criterion(self.discriminator(self.actor.encode_image(state_obs).detach()).squeeze(), obs_class) 
            d_loss = 0.5*(d_loss_rollout + d_loss_obs)
            
            self.discriminator_optimizer.zero_grad()
            d_loss.backward()
            self.discriminator_optimizer.step()
            
            encode_loss = criterion(self.discriminator(self.actor.encode_image(state_obs)).squeeze(), rollout_class) 
                
            self.actor_optimizer.zero_grad()
            loss = (-1) * (L_clip - self.c1 * L_vf + self.c2 * S - L_ia - L_ir - encode_loss).mean()
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
        

        
        
        
        
        
        

            
            
        
            
            
            

        