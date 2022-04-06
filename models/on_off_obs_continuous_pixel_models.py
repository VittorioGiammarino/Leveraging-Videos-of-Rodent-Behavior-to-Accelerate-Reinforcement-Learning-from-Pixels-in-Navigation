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
            
class SoftmaxActor(nn.Module):
    def __init__(self, state_dim, action_dim, use_memory = False, use_bn=True):
        super(SoftmaxActor, self).__init__()
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # Decide which components are enabled
        self.use_memory = use_memory
        self.in_channel = state_dim[2]
        n = state_dim[0]
        m = state_dim[1]
        print(f"image dim:{n}x{m}")

        if use_bn:
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, (2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(32),
            )
            
        else:
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 32, (2, 2)),
                nn.ReLU(),
            )
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*32
        print("Image embedding size: ", self.image_embedding_size)
        self.embedding = self.image_embedding_size
        
        # Define memory
        if self.use_memory:
            self.semi_memory_size = self.image_embedding_size
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
            self.embedding += self.semi_memory_size
        
        self.reg_layer = nn.Linear(self.embedding, 64)
        
        self.actor = nn.Sequential(
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

        # Define critic's model
        self.value_function = nn.Sequential(
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.inverse_model_action = nn.Sequential(
            nn.Linear(2*64, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim)
            )
        
        self.inverse_model_reward = nn.Sequential(
            nn.Linear(2*64, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Sigmoid()
            )
        
        self.lS = nn.Softmax(dim=1)
        
        # Initialize parameters correctly
        self.apply(init_params)
        
    def encode_image(self, state):
        x = state
        x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)
        bot = self.reg_layer(embedding)
        return bot
        
    def forward(self, state):
        embedding = self.encode_image(state)
        a = self.actor(embedding)
        return self.lS(torch.clamp(a,-10,10))
    
    def value_net(self, state):
        embedding = self.encode_image(state)
        q1 = self.value_function(embedding)   
        return q1
    
    def forward_inv_a(self, state, state_next):
        embedding = self.encode_image(state)
        embedding_next = self.encode_image(state_next)
        hh = torch.cat((embedding, embedding_next), dim=-1)
        a = self.inverse_model_action(hh)
        
        return self.lS(torch.clamp(a,-10,10))
    
    def forward_inv_reward(self, state, state_next):
        embedding = self.encode_image(state)
        embedding_next = self.encode_image(state_next)
        hh = torch.cat((embedding, embedding_next), dim=-1)
        r = self.inverse_model_reward(hh)
        
        return r
        
    def sample_inverse_model(self, state, state_next):
        prob = self.forward_inv_a(state, state_next)
        m = Categorical(prob)
        action = m.sample()
        
        return action    
    
    def sample(self, state):
        embedding = self.encode_image(state)
        
        self.log_Soft = nn.LogSoftmax(dim=1)
        a = self.actor(embedding)
        log_prob = self.log_Soft(torch.clamp(a,-10,10))
        
        prob = self.forward(state)
        m = Categorical(prob)
        action = m.sample()
        
        log_prob_sampled = log_prob.gather(1, action.reshape(-1,1).long())
        #log_prob_sampled = log_prob[torch.arange(len(action)),action]
        
        return action, log_prob_sampled
    
    def sample_log(self, state, action):
        embedding = self.encode_image(state)
        
        self.log_Soft = nn.LogSoftmax(dim=1)
        a = self.actor(embedding)
        log_prob = self.log_Soft(torch.clamp(a,-10,10))
                    
        log_prob_sampled = log_prob.gather(1, action.detach().reshape(-1,1).long()) # log_prob_sampled = log_prob[torch.arange(len(action)), action]
        
        return log_prob, log_prob_sampled.reshape(-1,1)
    
class Discriminator(nn.Module):
    def __init__(self, embedding_size = 64):
        super(Discriminator, self).__init__()

        # architecture
        self.discriminator = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
            )

    def forward(self, embedding):
        return torch.sigmoid(self.discriminator(embedding))
