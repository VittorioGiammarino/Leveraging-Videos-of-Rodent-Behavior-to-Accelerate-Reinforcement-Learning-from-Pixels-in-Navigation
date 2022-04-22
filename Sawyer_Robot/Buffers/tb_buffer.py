#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:58:58 2022

@author: vittoriogiammarino
"""
import numpy as np
import torch
import random

class GAE_TB_Buffer(object):
    def __init__(self, batch_size = 5, max_size=100):
        self.batch_size = batch_size
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.Buffer = {} 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i in range(self.max_size):
            self.Buffer['states'] = [[] for i in range(max_size)]
            self.Buffer['actions'] = [[] for i in range(max_size)]
            self.Buffer['rewards'] = [[] for i in range(max_size)]
            self.Buffer['returns'] = [[] for i in range(max_size)]
            self.Buffer['gammas'] = [[] for i in range(max_size)]
            self.Buffer['lambdas'] = [[] for i in range(max_size)]
            self.Buffer['episode_length'] = [[] for i in range(max_size)]
            
    def clear(self):
        self.Buffer['states'][self.ptr] = []
        self.Buffer['actions'][self.ptr] = []
        self.Buffer['rewards'][self.ptr] = []
        self.Buffer['returns'][self.ptr] = []
        self.Buffer['gammas'][self.ptr] = []
        self.Buffer['lambdas'][self.ptr] = []
        self.Buffer['episode_length'][self.ptr] = []
                
    def add(self, states, actions, rewards, returns, gammas, lambdas, episode_length):
        self.Buffer['states'][self.ptr] = states
        self.Buffer['actions'][self.ptr] = actions
        self.Buffer['rewards'][self.ptr] = rewards
        self.Buffer['returns'][self.ptr] = returns
        self.Buffer['gammas'][self.ptr] = gammas
        self.Buffer['lambdas'][self.ptr] = lambdas
        self.Buffer['episode_length'][self.ptr] = episode_length
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self):
        self.Buffer['advantage'] = [[] for i in range(self.batch_size)]
        ind = np.random.randint(0, self.size, size=self.batch_size)
        
        return (
                [(self.Buffer['states'][i]).to(self.device) for i in ind],
                [(self.Buffer['actions'][i]).to(self.device) for i in ind],
                [(self.Buffer['rewards'][i]).to(self.device) for i in ind],
                [(self.Buffer['returns'][i]).to(self.device) for i in ind],
                [(self.Buffer['gammas'][i]).to(self.device) for i in ind],
                [(self.Buffer['lambdas'][i]).to(self.device) for i in ind],
                [self.Buffer['episode_length'][i] for i in ind]
                )
        
    def add_Adv(self, i, advantage):
        self.Buffer['advantage'][i] = advantage