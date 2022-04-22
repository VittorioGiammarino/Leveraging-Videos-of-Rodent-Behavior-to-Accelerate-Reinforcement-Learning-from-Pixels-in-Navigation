#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:29:37 2021

@author: vittorio
"""
import numpy as np
import torch
import random

class ReplayBuffer(object):
    def __init__(self, action_space, state_dim, action_dim, max_size=int(1e5)):
        self.action_space = action_space
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim[2], state_dim[0], state_dim[1]), dtype=np.uint8)
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim[2], state_dim[0], state_dim[1]), dtype=np.uint8)
        self.reward = np.zeros((max_size, 1))
        self.cost = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(self, state, action, next_state, reward, cost, done):
        self.state[self.ptr] = state.transpose(2,0,1)
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state.transpose(2,0,1)
        self.reward[self.ptr] = reward
        self.cost[self.ptr] = cost
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        if self.action_space == "Discrete":
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
    			torch.LongTensor(self.action[ind]).to(self.device),
    			torch.FloatTensor(self.next_state[ind]).to(self.device),
    			torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.cost[ind]).to(self.device),
    			torch.FloatTensor(self.not_done[ind]).to(self.device)
                )
        elif self.action_space == "Continuous":
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
    			torch.FloatTensor(self.action[ind]).to(self.device),
    			torch.FloatTensor(self.next_state[ind]).to(self.device),
    			torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.cost[ind]).to(self.device),
    			torch.FloatTensor(self.not_done[ind]).to(self.device)
                )
        else:
            NotImplemented
    
    def convert_D4RL(self, dataset):
        self.state = dataset['observations'][:-1].transpose(0,3,1,2)
        self.next_state = dataset['next_observations'][:].transpose(0,3,1,2)
        self.action = dataset['actions'].reshape(-1,1)[:]
        self.reward = dataset['rewards'].reshape(-1,1)[:]
        self.not_done = 1. - dataset['terminals'].reshape(-1,1)[:]
        self.size = self.state.shape[0]
        
    def sample_trajectories(self, ntrajs):
        states = []
        actions = []
        rewards = []
        episode_length = []
        
        done_indexes = np.where(self.not_done==0)[0]
        max_ntrajs = len(done_indexes)
        
        ind = np.random.randint(0, max_ntrajs-1, size=ntrajs)
        
        if self.action_space == "Discrete":
            for i in range(ntrajs):
                states.append(torch.FloatTensor(self.state[done_indexes[ind[i]]:done_indexes[ind[i]+1]]).to(self.device))
                actions.append(torch.LongTensor(self.action[done_indexes[ind[i]]:done_indexes[ind[i]+1]]).to(self.device))
                rewards.append(torch.FloatTensor(self.reward[done_indexes[ind[i]]:done_indexes[ind[i]+1]]).to(self.device))
                episode_length.append(done_indexes[ind[i]+1]-done_indexes[ind[i]])
                
        elif self.action_space == "Continuous":
            for i in range(ntrajs):
                states.append(torch.FloatTensor(self.state[done_indexes[ind[i]]:done_indexes[ind[i]+1]]).to(self.device))
                actions.append(torch.FloatTensor(self.action[done_indexes[ind[i]]:done_indexes[ind[i]+1]]).to(self.device))
                rewards.append(torch.FloatTensor(self.reward[done_indexes[ind[i]]:done_indexes[ind[i]+1]]).to(self.device))
                episode_length.append(done_indexes[ind[i]+1]-done_indexes[ind[i]])
                
        else:
            NotImplemented
            
        return states, actions, rewards, episode_length
    
