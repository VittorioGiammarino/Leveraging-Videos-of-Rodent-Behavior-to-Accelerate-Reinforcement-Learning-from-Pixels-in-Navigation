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

class on_off_AWAC_Q_lambda_Peng:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, Entropy = True,   
                 num_steps_per_rollout=2000, l_rate_actor=3e-4, l_rate_alpha=3e-4, discount=0.99, tau=0.005, beta=3, 
                 gae_gamma = 0.99, gae_lambda = 0.99, minibatch_size=64, num_epochs=10, alpha=0.2, critic_freq=2):
        
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
        
        self.num_steps_per_rollout_on = num_steps_per_rollout
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.critic_freq = critic_freq
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        self.states = []
        self.actions = []
        self.target_Q = []
        self.advantage = []
        
        self.Entropy = Entropy
        self.target_entropy = -torch.FloatTensor([action_dim]).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr = l_rate_alpha) 
        self.alpha = alpha
        
        self.Total_t = 0
        self.Total_iter = 0
        self.total_it = 0
        
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
            
    def Q_lambda_Peng(self, env, args):
        step = 0
        self.Total_iter += 1
        self.states = []
        self.actions = []
        self.target_Q = []
        self.advantage = []

        while step < self.num_steps_per_rollout_on: 
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_gammas = []
            episode_lambdas = []    
            state, done = env.reset(), False
            t=0
            episode_reward = 0

            while not done and step < self.num_steps_per_rollout_on: 
            # Select action randomly or according to policy
                if self.Total_t < args.start_timesteps:
                    if args.action_space == "Continuous":
                        action = env.action_space.sample() 
                    elif args.action_space == "Discrete":
                        action = env.action_space.sample()  
                else:
                    action = on_off_AWAC_Q_lambda_Peng.select_action(self, state)
            
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
                
            episode_states = torch.FloatTensor(np.array(episode_states)).to(device)
            
            if self.action_space == "Discrete":
                episode_actions = torch.LongTensor(np.array(episode_actions)).to(device)
            elif self.action_space == "Continuous":
                episode_actions = torch.FloatTensor(np.array(episode_actions))
            
            episode_rewards = torch.FloatTensor(np.array(episode_rewards)).to(device)
            episode_gammas = torch.FloatTensor(np.array(episode_gammas)).to(device)
            episode_lambdas = torch.FloatTensor(np.array(episode_lambdas)).to(device)
            
            traj_size = t
            
            self.actor.eval()
            with torch.no_grad():
                
                pi_action, log_pi = self.actor.sample(episode_states)
                Q1_on, Q2_on = self.actor.critic_target(episode_states)
                
                if self.Entropy:
                    current_Q1 = Q1_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    current_Q2 = Q2_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    values = torch.min(current_Q1, current_Q2) - self.alpha*log_pi
                else:
                    current_Q1 = Q1_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    current_Q2 = Q2_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    values = torch.min(current_Q1, current_Q2)
                    
                final_bootstrap = values[-1].unsqueeze(-1)
                next_values = values[1:]
                next_action_values = torch.cat((episode_rewards[:-1].unsqueeze(-1) + self.gae_gamma*next_values, final_bootstrap))
                
                episode_adv = []
                episode_Q = []
                
                for j in range(traj_size):
                    off_policy_adjust = torch.cat((torch.FloatTensor([[0.]]).to(device), values[j+1:]))     
                    episode_deltas = next_action_values[j:] - off_policy_adjust
                    episode_Q.append(((episode_gammas*episode_lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum())
                    episode_adv.append(((episode_gammas*episode_lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum() - values[j])
                
                episode_advantage = torch.FloatTensor(episode_adv).to(device)
                episode_target_Q = torch.FloatTensor(episode_Q).to(device)
            
                self.advantage.append(episode_advantage)
                self.target_Q.append(episode_target_Q)
                
    def Q_lambda_Peng_off(self, replay_buffer, ntrajs):
        states = []
        actions = []
        target_Q = []
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
                episode_rewards = sampled_rewards[l] 
                episode_gammas = gammas_list[l]
                episode_lambdas = lambdas_list[l]
                
                traj_size = sampled_lengths[l] 
                
                self.actor.eval()
                
                pi_action, log_pi = self.actor.sample(episode_states)
                Q1_on, Q2_on = self.actor.critic_target(episode_states)
                
                if self.Entropy:
                    current_Q1 = Q1_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    current_Q2 = Q2_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    values = torch.min(current_Q1, current_Q2) - self.alpha*log_pi
                else:
                    current_Q1 = Q1_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    current_Q2 = Q2_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    values = torch.min(current_Q1, current_Q2)
                    
                final_bootstrap = values[-1].unsqueeze(-1)
                next_values = values[1:]
                next_action_values = torch.cat((episode_rewards[:-1] + self.gae_gamma*next_values, final_bootstrap))
                
                episode_adv = []
                episode_Q = []
                
                for j in range(traj_size):
                    off_policy_adjust = torch.cat((torch.FloatTensor([[0.]]).to(device), values[j+1:]))     
                    episode_deltas = next_action_values[j:] - off_policy_adjust
                    episode_Q.append(((episode_gammas*episode_lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum())
                    episode_adv.append(((episode_gammas*episode_lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum() - values[j])
                
                episode_advantage = torch.FloatTensor(episode_adv).to(device)
                episode_target_Q = torch.FloatTensor(episode_Q).to(device)
            
                states.append(episode_states)
                actions.append(episode_actions)
                target_Q.append(episode_target_Q)
                advantage.append(episode_advantage)
                
        return states, actions, target_Q, advantage
    
    def train(self, states_off, actions_off, target_Q_off, advantage_off):
        self.total_it += 1
        
        states_on = torch.FloatTensor(np.array(self.states)).to(device)
        
        if self.action_space == "Discrete":
            actions_on = torch.LongTensor(np.array(self.actions)).to(device)
        elif self.action_space == "Continuous":
            actions_on = torch.FloatTensor(np.array(self.actions)).to(device)
        
        target_Q_on = torch.cat(self.target_Q)
        advantage_on = torch.cat(self.advantage).to(device)
        
        states_off = torch.cat(states_off)
        rollout_states = torch.cat((states_on, states_off))
        
        actions_off = torch.cat(actions_off)
        rollout_actions = torch.cat((actions_on, actions_off.squeeze()))
        
        target_Q_off = torch.cat(target_Q_off)
        rollout_target_Q = torch.cat((target_Q_on, target_Q_off))
        
        advantage_off = torch.cat(advantage_off)
        rollout_advantage = torch.cat((advantage_on, advantage_off))
        
        rollout_advantage = (rollout_advantage-rollout_advantage.mean())/(rollout_advantage.std()+1e-6)
        
        self.actor.train()
        self.num_steps_per_rollout = len(rollout_advantage)
        max_steps = self.num_epochs * (self.num_steps_per_rollout // self.minibatch_size)
        
        for _ in range(max_steps):
            
            minibatch_indices = np.random.choice(range(self.num_steps_per_rollout), self.minibatch_size, False)
            batch_states=rollout_states[minibatch_indices]
            batch_actions = rollout_actions[minibatch_indices]
            batch_target_Q = rollout_target_Q[minibatch_indices]
            batch_advantage = rollout_advantage[minibatch_indices]
                    
            if self.action_space == "Discrete":
                log_prob, log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)

            elif self.action_space == "Continuous": 
                log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)
                
            r = (log_prob_rollout).squeeze()
            weights = F.softmax(batch_advantage/self.beta, dim=0).detach()
            L_clip = r*weights
            
            if self.action_space == "Discrete":
                Q1, Q2 = self.actor.critic_net(batch_states)
                current_Q1 = Q1.gather(1, batch_actions.long().unsqueeze(-1)) 
                current_Q2 = Q2.gather(1, batch_actions.long().unsqueeze(-1)) 
            
            elif self.action_space == "Continuous":
                #current Q estimates
                current_Q1, current_Q2 = self.critic(batch_states, batch_actions)
            
            critic_loss = F.mse_loss(current_Q1.squeeze(), batch_target_Q) + F.mse_loss(current_Q1.squeeze(), batch_target_Q)
                
            self.actor_optimizer.zero_grad()
            
            if self.Entropy:
                _, log_pi_state = self.actor.sample(batch_states)
                loss = (-1) * (L_clip - self.alpha*log_pi_state).mean() + critic_loss
            else:
                loss = (-1) * (L_clip).mean() + critic_loss
            
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
        torch.save(self.value_function.state_dict(), filename + "_value_function")
        torch.save(self.value_function_optimizer.state_dict(), filename + "_value_function_optimizer")
    
    def load_critic(self, filename):
        self.value_function.load_state_dict(torch.load(filename + "_value_function"))      
        self.value_function_optimizer.load_state_dict(torch.load(filename + "_value_function_optimizer")) 
        

        
        
        
        
        
        

            
            
        
            
            
            

        