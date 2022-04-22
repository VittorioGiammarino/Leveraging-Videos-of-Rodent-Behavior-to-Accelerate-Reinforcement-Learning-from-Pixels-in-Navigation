#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:55:49 2021

@author: vittorio
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

class SoftmaxHierarchicalActor:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim, input_size = 64):
            super(SoftmaxHierarchicalActor.NN_PI_LO, self).__init__()
            
            self.action_dim = action_dim
            self.state_dim = state_dim
            
            self.actor = nn.Sequential(
                nn.Tanh(),
                nn.Linear(input_size, action_dim)
            )

            # Define critic's model
            self.value_function = nn.Sequential(
                nn.Tanh(),
                nn.Linear(input_size, 1)
            )
            
            # Define critic's model
            self.Q1 = nn.Sequential(
                nn.Tanh(),
                nn.Linear(input_size, action_dim)
            )
            
            # Define critic's model
            self.Q2 = nn.Sequential(
                nn.Tanh(),
                nn.Linear(input_size, action_dim)
            )
            
            self.critic_target_Q1 = copy.deepcopy(self.Q1)
            self.critic_target_Q2 = copy.deepcopy(self.Q2)
            self.actor_target = copy.deepcopy(self.actor)
            
            self.lS = nn.Softmax(dim=1)
            
            # Initialize parameters correctly
            self.apply(init_params)
                        
        def forward(self, embedding):
            a = self.actor(embedding)
            return self.lS(torch.clamp(a,-10,10))
        
        def actor_target_net(self, embedding):
            a = self.actor_target(embedding)
            return self.lS(torch.clamp(a,-10,10))
        
        def value_net(self, embedding):
            q1 = self.value_function(embedding)   
            return q1
        
        def critic_net(self, embedding):
            q1 = self.Q1(embedding)   
            q2 = self.Q2(embedding)
            return q1, q2
        
        def critic_target(self, embedding):
            q1 = self.critic_target_Q1(embedding)   
            q2 = self.critic_target_Q2(embedding)
            return q1, q2
        
        def sample(self, embedding):
            self.log_Soft = nn.LogSoftmax(dim=1)
            a = self.actor(embedding)
            log_prob = self.log_Soft(torch.clamp(a,-10,10))
            
            prob = self.forward(embedding)
            m = Categorical(prob)
            action = m.sample()
            
            log_prob_sampled = log_prob.gather(1, action.reshape(-1,1).long())
            #log_prob_sampled = log_prob[torch.arange(len(action)),action]
            
            return action, log_prob_sampled
        
        def sample_target(self, embedding):            
            self.log_Soft = nn.LogSoftmax(dim=1)
            a = self.actor_target(embedding)
            log_prob = self.log_Soft(torch.clamp(a,-10,10))
            
            prob = self.actor_target_net(embedding)
            m = Categorical(prob)
            action = m.sample()
            
            log_prob_sampled = log_prob.gather(1, action.reshape(-1,1).long())
            #log_prob_sampled = log_prob[torch.arange(len(action)),action]
            
            return action, log_prob_sampled
        
        def sample_log(self, embedding, action):            
            self.log_Soft = nn.LogSoftmax(dim=1)
            a = self.actor(embedding)
            log_prob = self.log_Soft(torch.clamp(a,-10,10))
                        
            log_prob_sampled = log_prob.gather(1, action.detach().reshape(-1,1).long()) # log_prob_sampled = log_prob[torch.arange(len(action)), action]
            
            return log_prob, log_prob_sampled.reshape(-1,1)
            
        
    class NN_PI_B(nn.Module):
        def __init__(self, state_dim, termination_dim):
            super(SoftmaxHierarchicalActor.NN_PI_B, self).__init__()
            
            self.l1 = nn.Linear(state_dim,10)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(10,10)
            self.l3 = nn.Linear(10,termination_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            b = self.l1(state)
            b = F.relu(self.l2(b))
            return self.lS(torch.clamp(self.l3(b),-10,10))            
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            b = self.l1(state)
            b = F.relu(self.l2(b))
            log_prob = self.log_Soft(torch.clamp(self.l3(b),-10,10))
            
            prob = self.forward(state)
            m = Categorical(prob)
            termination = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(termination)), termination]
            
            return termination, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, termination):
            self.log_Soft = nn.LogSoftmax(dim=1)
            b = self.l1(state)
            b = F.relu(self.l2(b))
            log_prob = self.log_Soft(torch.clamp(self.l3(b),-10,10))
                        
            log_prob_sampled = log_prob[torch.arange(len(termination)), termination]
            
            return log_prob, log_prob_sampled.reshape(-1,1)
        
    class NN_PI_HI(nn.Module):
        def __init__(self, state_dim, option_dim):
            super(SoftmaxHierarchicalActor.NN_PI_HI, self).__init__()
            
            self.l1 = nn.Linear(state_dim,5)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(5,5)
            self.l3 = nn.Linear(5,option_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)

        def forward(self, state):
            o = self.l1(state)
            o = F.relu(self.l2(o))
            return self.lS(torch.clamp(self.l3(o),-10,10))
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            o = self.l1(state)
            o = F.relu(self.l2(o))
            log_prob = self.log_Soft(torch.clamp(self.l3(o),-10,10))
            
            prob = self.forward(state)
            m = Categorical(prob)
            option = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(option)), option]
            
            return option, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, option):
            self.log_Soft = nn.LogSoftmax(dim=1)
            o = self.l1(state)
            o = F.relu(self.l2(o))
            log_prob = self.log_Soft(torch.clamp(self.l3(o),-10,10))
                        
            log_prob_sampled = log_prob[torch.arange(len(option)), option]
            
            return log_prob, log_prob_sampled.reshape(-1,1)       
  
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, option_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim + option_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim + option_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action, option):
		sao = torch.cat([state, action, option], 1)

		q1 = F.relu(self.l1(sao))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sao))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action, option):
		sao = torch.cat([state, action, option], 1)

		q1 = F.relu(self.l1(sao))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1
    
class Critic_discrete(nn.Module):
	def __init__(self, state_dim, action_cardinality, option_dim):
		super(Critic_discrete, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + option_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_cardinality)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + option_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, action_cardinality)


	def forward(self, state, option):
		so = torch.cat([state, option], 1)

		q1 = F.relu(self.l1(so))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(so))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, option):
		so = torch.cat([state, option], 1)

		q1 = F.relu(self.l1(so))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1
    
class Critic_flat(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_flat, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
class Critic_flat_discrete(nn.Module):
    def __init__(self, state_dim, action_cardinality, vision_embedding = 1024, use_memory=False):
        super(Critic_flat_discrete, self).__init__()
        
        # Decide which components are enabled
        self.vision_embedding = vision_embedding
        self.use_memory = use_memory
        self.semi_memory_size = vision_embedding
        
        self.in_channel = state_dim[2]

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(self.in_channel, 16, kernel_size = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(16, 32, kernel_size = 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 2),
            nn.ReLU(),
        )
        
        n = state_dim[0]
        m = state_dim[1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, vision_embedding)
        self.embedding = vision_embedding
        
        # Define memory
        if self.use_memory:
            self.semi_memory_size = vision_embedding
            self.memory_rnn = nn.LSTMCell(self.vision_embedding, self.semi_memory_size)
            self.embedding += self.semi_memory_size
        

        # Q1 architecture
        self.l1 = nn.Linear(self.embedding, 2*self.embedding)
        self.l2 = nn.Linear(2*self.embedding, 2*self.embedding)
        self.l3 = nn.Linear(2*self.embedding, action_cardinality)

        # Q2 architecture
        self.l4 = nn.Linear(self.embedding, 2*self.embedding)
        self.l5 = nn.Linear(2*self.embedding, 2*self.embedding)
        self.l6 = nn.Linear(2*self.embedding, action_cardinality)
        
        # Initialize parameters correctly
        self.apply(init_params)


    def forward(self, state):
        image_embedding = self.image_conv(state)
        x = self.avgpool(image_embedding)
        x = x.reshape(x.shape[0],-1)
        embedding = self.fc(x)
        
        q1 = F.relu(self.l1(embedding))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(embedding))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2


    def Q1(self, state):
        image_embedding = self.image_conv(state)
        x = self.avgpool(image_embedding)
        x = x.reshape(x.shape[0],-1)
        embedding = self.fc(x)
        
        q1 = F.relu(self.l1(embedding))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1
    
class Value_net_CNN(nn.Module):
    def __init__(self, state_dim, vision_embedding = 1024, use_memory=False):
        super(Value_net_CNN, self).__init__()
        
        # Decide which components are enabled
        self.vision_embedding = vision_embedding
        self.use_memory = use_memory
        self.semi_memory_size = vision_embedding
        
        self.in_channel = state_dim[2]

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(self.in_channel, 16, kernel_size = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(16, 32, kernel_size = 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 2),
            nn.ReLU(),
        )
        
        n = state_dim[0]
        m = state_dim[1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, vision_embedding)
        self.embedding = vision_embedding
        
        # Define memory
        if self.use_memory:
            self.semi_memory_size = vision_embedding
            self.memory_rnn = nn.LSTMCell(self.vision_embedding, self.semi_memory_size)
            self.embedding += self.semi_memory_size
        
        
        # Value_net architecture
        self.l1 = nn.Linear(self.embedding, 4*self.embedding)
        self.l2 = nn.Linear(4*self.embedding, 4*self.embedding)
        self.l3 = nn.Linear(4*self.embedding, 1)
        
        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, state):
        
        image_embedding = self.image_conv(state)
        x = self.avgpool(image_embedding)
        x = x.reshape(x.shape[0],-1)
        embedding = self.fc(x)
        
        q1 = F.relu(self.l1(embedding))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)    
        
        return q1
    
class Value_net_H(nn.Module):
    def __init__(self, state_dim, option_dim):
        super(Value_net_H, self).__init__()
        # Value_net architecture
        self.l1 = nn.Linear(state_dim + option_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, option):
        so = torch.cat([state, option], 1)
        
        q1 = F.relu(self.l1(so))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)    
        return q1
    
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        # architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        return torch.sigmoid(self.get_logits(state, action))

    def get_logits(self, state, action):
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        sa = torch.cat([state, action], 1)
        d = F.relu(self.l1(sa))
        d = F.relu(self.l2(d))
        d = self.l3(d)
        return d