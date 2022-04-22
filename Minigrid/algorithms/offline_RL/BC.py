#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:02:13 2022

@author: vittoriogiammarino
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from models.simple_minigird_models import SoftmaxHierarchicalActor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BC(object):
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, l_rate_actor=3e-4, tau=0.005):

        self.actor = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_space_cardinality).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
        self.action_space = "Discrete"
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.action_space_cardinality = action_space_cardinality

        self.max_action = max_action
        self.action_dim = action_dim
        self.tau = tau

    def select_action(self, state):
        return self.select_action_cloning(state)       

    def select_action_cloning(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.transpose(2,0,1)).unsqueeze(0).to(device)
            action, _ = self.actor.sample_target(state)
            return int((action).cpu().data.numpy().flatten())

    def train(self, replay_buffer, batch_size=256):
        
        # Sample replay buffer 
        state, action, next_state, reward, cost, not_done = replay_buffer.sample(batch_size)

        action_prob = self.actor(state)  
        m = F.one_hot(action.squeeze().cpu(), self.action_space_cardinality).float().to(device)
        recon_loss = F.mse_loss(action_prob, m)

        self.actor_optimizer.zero_grad()
        recon_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.actor.parameters(), self.actor.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def save_actor(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        
    def load_actor(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))