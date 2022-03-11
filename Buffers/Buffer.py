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
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim[2], state_dim[0], state_dim[1]))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim[2], state_dim[0], state_dim[1]))
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
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.cost[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
            )
    
class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"

class PrioritizedReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e5), eps=1e-2, alpha=0.1, beta=0.1):
        
        self.tree = SumTree(size=max_size)
        
        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps        
        
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim[2], state_dim[0], state_dim[1]))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim[2], state_dim[0], state_dim[1]))
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
        
        self.tree.add(self.max_priority, self.ptr)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()
        
        batch = (
            torch.FloatTensor(self.state[sample_idxs]).to(self.device),
			torch.FloatTensor(self.action[sample_idxs]).to(self.device),
			torch.FloatTensor(self.next_state[sample_idxs]).to(self.device),
			torch.FloatTensor(self.reward[sample_idxs]).to(self.device),
            torch.FloatTensor(self.cost[sample_idxs]).to(self.device),
			torch.FloatTensor(self.not_done[sample_idxs]).to(self.device)
            )
        
        return batch, weights.to(self.device), tree_idxs
    
    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy().flatten()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
                

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
                [self.Buffer['states'][i] for i in ind],
                [self.Buffer['actions'][i] for i in ind],
                [self.Buffer['rewards'][i] for i in ind],
                [self.Buffer['returns'][i] for i in ind],
                [self.Buffer['gammas'][i] for i in ind],
                [self.Buffer['lambdas'][i] for i in ind],
                [self.Buffer['episode_length'][i] for i in ind]
                )
        
    def add_Adv(self, i, advantage):
        self.Buffer['advantage'][i] = advantage
        
class H_V_trace_Buffer(object):
    def __init__(self, num_options, max_size=4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.num_options = num_options
        
        self.Buffer = [{} for i in range(max_size)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i in range(self.max_size):
            self.Buffer[i]['states'] = []
            self.Buffer[i]['actions'] = []
            self.Buffer[i]['options'] = []
            self.Buffer[i]['terminations'] = []
            self.Buffer[i]['rewards'] = []
            self.Buffer[i]['returns'] = []
            self.Buffer[i]['gammas'] = []
            self.Buffer[i]['episode_length'] = []
            self.Buffer[i]['advantage'] = []
            self.Buffer[i]['advantage_option'] = []
            
            self.Buffer[i]['log_pi_hi_old'] = []
            for option in range(num_options):
                self.Buffer[i][f'log_pi_lo_old_{option}'] = []
                self.Buffer[i][f'log_pi_b_old_{option}'] = []
            
    def clear(self):
        self.Buffer[self.ptr]['states'] = []
        self.Buffer[self.ptr]['actions'] = []
        self.Buffer[self.ptr]['options'] = []
        self.Buffer[self.ptr]['terminations'] = []
        self.Buffer[self.ptr]['rewards'] = []
        self.Buffer[self.ptr]['returns'] = []
        self.Buffer[self.ptr]['gammas'] = []
        self.Buffer[self.ptr]['episode_length'] = []
        
        self.Buffer[self.ptr]['log_pi_hi_old'] = []
        for option in range(self.num_options):
            self.Buffer[self.ptr][f'log_pi_lo_old_{option}'] = []
            self.Buffer[self.ptr][f'log_pi_b_old_{option}'] = []
                
    def add(self, states, actions, options, terminations, rewards, returns, gammas, log_pi_lo_old, log_pi_b_old, log_pi_hi_old, episode_length):
        self.Buffer[self.ptr]['states'].append(states)
        self.Buffer[self.ptr]['actions'].append(actions)
        self.Buffer[self.ptr]['options'].append(options)
        self.Buffer[self.ptr]['terminations'].append(terminations)
        self.Buffer[self.ptr]['rewards'].append(rewards)
        self.Buffer[self.ptr]['returns'].append(returns)
        self.Buffer[self.ptr]['gammas'].append(gammas)
        self.Buffer[self.ptr]['episode_length'].append(episode_length)
        
        self.Buffer[self.ptr]['log_pi_hi_old'].append(log_pi_hi_old)
        for option in range(self.num_options):
            self.Buffer[self.ptr][f'log_pi_lo_old_{option}'].append(log_pi_lo_old[option])
            self.Buffer[self.ptr][f'log_pi_b_old_{option}'].append(log_pi_b_old[option])       
        
    def clear_Adv(self, i):
        self.Buffer[i]['advantage'] = []
        self.Buffer[i]['advantage_option'] = []
        
    def add_Adv(self, i, advantage, advantage_option):
        self.Buffer[i]['advantage'].append(advantage)
        self.Buffer[i]['advantage_option'].append(advantage_option)
        
    def update_counters(self):
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)