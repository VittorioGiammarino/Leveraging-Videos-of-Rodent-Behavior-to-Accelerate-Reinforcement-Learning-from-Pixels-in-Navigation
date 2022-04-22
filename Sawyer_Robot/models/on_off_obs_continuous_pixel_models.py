#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:50:28 2022

@author: vittoriogiammarino
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
            
class TanhGaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_bn=True):
        super(TanhGaussianActor, self).__init__()
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
        self.high = max_action
        self.low = -max_action
        
        self.actor = nn.Sequential(
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

        # Define critic's model
        self.value_function = nn.Sequential(
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Define critic's model
        self.Q1 = nn.Sequential(
            nn.Linear(64+action_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Define critic's model
        self.Q2 = nn.Sequential(
            nn.Linear(64+action_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self.critic_target_Q1 = copy.deepcopy(self.Q1)
        self.critic_target_Q2 = copy.deepcopy(self.Q2)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.inverse_model_action = nn.Sequential(
            nn.Linear(2*64, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim)
            )
        self.log_std_inverse_model_action = torch.nn.Parameter(torch.zeros(action_dim))
        
        self.inverse_model_reward = nn.Sequential(
            nn.Linear(2*64, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Sigmoid()
            )
                
        # Initialize parameters correctly
        self.apply(init_params)
                
    def forward(self, embedding):
        mean = self.actor(embedding)
        log_std = self.log_std.clamp(-20,2)
        std = torch.exp(log_std)
        return mean, std
    
    def unsquash(self, values):
        normed_values = (values - self.low[0])/(self.high[0] - self.low[0])*2.0 - 1.0
        stable_normed_values = torch.clamp(normed_values, -1+1e-4, 1-1e-4)
        unsquashed = torch.atanh(stable_normed_values)
        return unsquashed.float()
    
    def sample_log(self, embedding, action):
        mean, std = self.forward(embedding)
        normal = torch.distributions.Normal(mean, std)
        x = self.unsquash(action)
        y = torch.tanh(x) 
        log_prob = torch.clamp(normal.log_prob(x), -5, 5)
        log_prob -= torch.log((1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob
    
    def sample(self, embedding):
        mean, std = self.forward(embedding)
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)        
        action = y*self.high[0]
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.high[0]*(1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)*self.high[0]
        return action, log_prob, mean 
    
    def Distb(self, embedding):
        mean = self.actor(embedding)
        log_std = self.log_std.clamp(-20,2)
        std = torch.exp(log_std)
        cov_mtx = torch.eye(self.action_dim).to(device) * (std ** 2)
        distb = torch.distributions.MultivariateNormal(mean, cov_mtx)
        return distb
    
    def value_net(self, embedding):
        q1 = self.value_function(embedding)   
        return q1
    
    def critic_net(self, embedding, action):
        sa = torch.cat([embedding, action], 1)
        q1 = self.Q1(sa)   
        q2 = self.Q2(sa)
        return q1, q2
    
    def critic_target(self, embedding, action):
        sa = torch.cat([embedding, action], 1)
        q1 = self.critic_target_Q1(sa)   
        q2 = self.critic_target_Q2(sa)
        return q1, q2
    
    def forward_inv_a(self, embedding, embedding_next):
        hh = torch.cat((embedding, embedding_next), dim=-1)
        mean = self.inverse_model_action(hh)
        log_std = self.log_std_inverse_model_action.clamp(-20,2)
        std = torch.exp(log_std)
        return mean, std
    
    def forward_inv_reward(self, embedding, embedding_next):
        hh = torch.cat((embedding, embedding_next), dim=-1)
        r = self.inverse_model_reward(hh)
        return r
        
    def sample_inverse_model(self, embedding, embedding_next):
        mean, std = self.forward_inv_a(embedding, embedding_next)
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)        
        action = y*self.high[0]

        return action

class Discriminator_GAN_Sawyer(nn.Module):
    def __init__(self, embedding_size = 64):
        super(Discriminator_GAN_Sawyer, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        

    def forward(self, embedding):
        return self.discriminator(embedding)
    
class Discriminator_WGAN_Sawyer(nn.Module):
    def __init__(self, embedding_size = 64):
        super(Discriminator_WGAN_Sawyer, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )
        

    def forward(self, embedding):
        return self.discriminator(embedding)   

class Encoder_Sawyer(nn.Module):
    def __init__(self, state_dim, action_dim, 
                 class_latent_size = 8, content_latent_size = 64, input_channel = 3, flatten_size = 256):
        super(Encoder_Sawyer, self).__init__()

        self.class_latent_size = class_latent_size
        self.content_latent_size = content_latent_size
        self.flatten_size = flatten_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(flatten_size, content_latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, content_latent_size)
        self.linear_classcode = nn.Linear(flatten_size, class_latent_size) 
        
    def forward(self, state):
        x = state/255
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)

        return mu    

