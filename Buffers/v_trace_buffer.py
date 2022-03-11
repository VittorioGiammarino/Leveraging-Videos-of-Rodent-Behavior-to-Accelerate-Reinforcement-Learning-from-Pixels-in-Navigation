#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:57:47 2022

@author: vittoriogiammarino
"""

import numpy as np
import torch
import random

class V_trace_Buffer(object):
    def __init__(self, max_size=4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.Buffer = [{} for i in range(max_size)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i in range(self.max_size):
            self.Buffer[i]['states'] = []
            self.Buffer[i]['actions'] = []
            self.Buffer[i]['rewards'] = []
            self.Buffer[i]['returns'] = []
            self.Buffer[i]['gammas'] = []
            self.Buffer[i]['log_pi_old'] = []
            self.Buffer[i]['episode_length'] = []
            self.Buffer[i]['advantage'] = []
            
    def clear(self):
        self.Buffer[self.ptr]['states'] = []
        self.Buffer[self.ptr]['actions'] = []
        self.Buffer[self.ptr]['rewards'] = []
        self.Buffer[self.ptr]['returns'] = []
        self.Buffer[self.ptr]['gammas'] = []
        self.Buffer[self.ptr]['log_pi_old'] = []
        self.Buffer[self.ptr]['episode_length'] = []
                
    def add(self, states, actions, rewards, returns, gammas, log_pi_old, episode_length):
        self.Buffer[self.ptr]['states'].append(states)
        self.Buffer[self.ptr]['actions'].append(actions)
        self.Buffer[self.ptr]['rewards'].append(rewards)
        self.Buffer[self.ptr]['returns'].append(returns)
        self.Buffer[self.ptr]['gammas'].append(gammas)
        self.Buffer[self.ptr]['log_pi_old'].append(log_pi_old)
        self.Buffer[self.ptr]['episode_length'].append(episode_length)
        
    def clear_Adv(self, i):
        self.Buffer[i]['advantage'] = []
        
    def add_Adv(self, i, advantage):
        self.Buffer[i]['advantage'].append(advantage)
        
    def update_counters(self):
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)