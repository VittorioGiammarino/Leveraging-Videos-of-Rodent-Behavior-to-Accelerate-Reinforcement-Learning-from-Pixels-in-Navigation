#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:57:49 2022

@author: vittoriogiammarino
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from models import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
    
class BasicBlock(nn.Module):    
    
    expansion: int = 1
    
    def __init__(self, input_dim, num_features, identity_downsample = None):
        super(BasicBlock, self).__init__()
        
        self.l1 = nn.Linear(input_dim, num_features)
        self.l2 = nn.Linear(num_features, num_features)
        self.bn = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.l1(x))
        x = self.l2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        x = self.bn(x)
        
        return x

class SoftmaxHierarchicalActorMLP:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim, vision_embedding=128, use_memory=False):
            super(SoftmaxHierarchicalActorMLP.NN_PI_LO, self).__init__()
            
            # Decide which components are enabled
            self.vision_embedding = vision_embedding
            self.use_memory = use_memory
            self.semi_memory_size = vision_embedding
            
            self.in_channel = state_dim[2]
            self.vision_encoder = ResNet.ResNet18(self.in_channel, vision_embedding).to(device)
            
            self.input_dim = vision_embedding
            self.action_dim = action_dim
            
            self.int_dim = vision_embedding
            
            self.block = BasicBlock
            self.layers = [2, 2]
            
            self.layer1 = self._make_layer(self.block, self.layers[0], num_features=256)
            self.layer2 = self._make_layer(self.block, self.layers[1], num_features=512)
            self.fc = nn.Linear(512, action_dim)
            
            self.lS = nn.Softmax(dim=1)
            
            # Initialize parameters correctly
            self.apply(init_params)
            
            
        def _make_layer(self, block, num_residual_blocks, num_features):
            identity_downsample = None
            layers = []
    
            if self.int_dim != num_features * block.expansion:
                identity_downsample = nn.Sequential(
                    nn.Linear(self.int_dim, num_features * block.expansion),
                    nn.BatchNorm1d(num_features * block.expansion),
                )
    
            layers.append(block(self.int_dim, num_features * block.expansion, identity_downsample))
            self.int_dim = num_features * block.expansion
            
            for _ in range(num_residual_blocks - 1):
                layers.append(block(self.int_dim, num_features))
    
            return nn.Sequential(*layers)
        		
        def forward(self, state):     
            s = self.vision_encoder(state)

            x = self.layer1(s)
            x = self.layer2(x)
            score = self.fc(x)

            return self.lS(torch.clamp(score,-10,10))
        
        def sample(self, state):
            s = self.vision_encoder(state)

            x = self.layer1(s)
            x = self.layer2(x)
            score = self.fc(x)
            
            self.log_Soft = nn.LogSoftmax(dim=1)
            log_prob = self.log_Soft(torch.clamp(score,-10,10))
            
            prob = self.forward(state)
            m = Categorical(prob)
            action = m.sample()
            
            log_prob_sampled = log_prob.gather(1, action.reshape(-1,1).long())
            #log_prob_sampled = log_prob[torch.arange(len(action)),action]
            
            return action, log_prob_sampled
        
        def sample_log(self, state, action):
            s = self.vision_encoder(state)

            x = self.layer1(s)
            x = self.layer2(x)
            score = self.fc(x)
            
            self.log_Soft = nn.LogSoftmax(dim=1)
            log_prob = self.log_Soft(torch.clamp(score,-10,10))
                        
            log_prob_sampled = log_prob.gather(1, action.detach().reshape(-1,1).long()) # log_prob_sampled = log_prob[torch.arange(len(action)), action]
            
            return log_prob, log_prob_sampled.reshape(-1,1)
        
class CriticMLP_flat_discrete(nn.Module):
    def __init__(self, state_dim, action_cardinality, vision_embedding=128, use_memory=False):
        super(CriticMLP_flat_discrete, self).__init__()
        
        # Decide which components are enabled
        self.vision_embedding = vision_embedding
        self.use_memory = use_memory
        self.semi_memory_size = vision_embedding
        
        self.in_channel = state_dim[2]
        self.vision_encoder = ResNet.ResNet18(self.in_channel, vision_embedding).to(device)
        
        self.input_dim = vision_embedding
        self.action_dim = action_cardinality
        
        self.block = BasicBlock
        self.layers = [2, 2]
        
        self.q1_int_dim = vision_embedding
        self.q1_layer1 = self._make_layerQ1(self.block, self.layers[0], num_features=256)
        self.q1_layer2 = self._make_layerQ1(self.block, self.layers[1], num_features=512)
        self.q1_fc = nn.Linear(512, action_cardinality)
        
        self.q2_int_dim = vision_embedding
        self.q2_layer1 = self._make_layerQ2(self.block, self.layers[0], num_features=256)
        self.q2_layer2 = self._make_layerQ2(self.block, self.layers[1], num_features=512)
        self.q2_fc = nn.Linear(512, action_cardinality)
        
        # Initialize parameters correctly
        self.apply(init_params)
        
    def _make_layerQ1(self, block, num_residual_blocks, num_features):
        identity_downsample = None
        layers = []

        if self.q1_int_dim != num_features * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Linear(self.q1_int_dim, num_features * block.expansion),
                nn.BatchNorm1d(num_features * block.expansion),
            )

        layers.append(block(self.q1_int_dim, num_features * block.expansion, identity_downsample))
        self.q1_int_dim = num_features * block.expansion
        
        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.q1_int_dim, num_features))

        return nn.Sequential(*layers)
        
    def _make_layerQ2(self, block, num_residual_blocks, num_features):
        identity_downsample = None
        layers = []

        if self.q2_int_dim != num_features * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Linear(self.q2_int_dim, num_features * block.expansion),
                nn.BatchNorm1d(num_features * block.expansion),
            )

        layers.append(block(self.q2_int_dim, num_features * block.expansion, identity_downsample))
        self.q2_int_dim = num_features * block.expansion
        
        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.q2_int_dim, num_features))

        return nn.Sequential(*layers)

    def forward(self, state):

        s = self.vision_encoder(state)
        
        q1 = self.q1_layer1(s)
        q1 = self.q1_layer2(q1)
        q1 = self.q1_fc(q1)
        
        q2 = self.q2_layer1(s)
        q2 = self.q2_layer2(q2)
        q2 = self.q1_fc(q2)
        
        return q1, q2

    def Q1(self, state):
        s = self.vision_encoder(state)
        
        q1 = self.q1_layer1(s)
        q1 = self.q1_layer2(q1)
        q1 = self.q1_fc(q1)
        
        return q1
    
class ValueNetMLP(nn.Module):
    def __init__(self, state_dim, vision_embedding=128, use_memory=False):
        super(ValueNetMLP, self).__init__()
        
        # Decide which components are enabled
        self.vision_embedding = vision_embedding
        self.use_memory = use_memory
        self.semi_memory_size = vision_embedding
        
        self.in_channel = state_dim[2]
        self.vision_encoder = ResNet.ResNet18(self.in_channel, vision_embedding).to(device)
        
        self.input_dim = vision_embedding
        
        self.block = BasicBlock
        self.layers = [2, 2]
        
        self.q1_int_dim = vision_embedding
        self.q1_layer1 = self._make_layerQ1(self.block, self.layers[0], num_features=256)
        self.q1_layer2 = self._make_layerQ1(self.block, self.layers[1], num_features=512)
        self.q1_fc = nn.Linear(512, 1)
        
        # Initialize parameters correctly
        self.apply(init_params)
        
    def _make_layerQ1(self, block, num_residual_blocks, num_features):
        identity_downsample = None
        layers = []

        if self.q1_int_dim != num_features * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Linear(self.q1_int_dim, num_features * block.expansion),
                nn.BatchNorm1d(num_features * block.expansion),
            )

        layers.append(block(self.q1_int_dim, num_features * block.expansion, identity_downsample))
        self.q1_int_dim = num_features * block.expansion
        
        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.q1_int_dim, num_features))

        return nn.Sequential(*layers)

    def forward(self, state):

        s = self.vision_encoder(state)
        
        q1 = self.q1_layer1(s)
        q1 = self.q1_layer2(q1)
        q1 = self.q1_fc(q1)
        
        return q1

