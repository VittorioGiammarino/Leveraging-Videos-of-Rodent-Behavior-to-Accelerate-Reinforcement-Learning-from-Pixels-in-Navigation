#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 17:44:56 2021

@author: vittorio
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as ptch
import pickle

def load_obj(name):
    with open('specs/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# environments = ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'LunarLander-v2', 'LunarLanderContinuous-v2',
#                 'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 'Humanoid-v3', 'HumanoidStandup-v2', 'Swimmer-v3', 'Walker2d-v3',
#                 'FetchPickAndPlace-v1', 'FetchPush-v1', 'FetchReach-v1', 'FetchSlide-v1', 'HandManipulateBlock-v0',
#                 'HandManipulateEgg-v0', 'HandManipulatePen-v0', 'HandReach-v0']

environments = ['BipedalWalker-v3', 'LunarLander-v2', 'LunarLanderContinuous-v2', 'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 
                'Humanoid-v3', 'HumanoidStandup-v2', 'Swimmer-v3', 'Walker2d-v3', 'BipedalWalkerHardcore-v3']

environments = ['BipedalWalkerHardcore-v3']

modes = ['RL', 'HRL']
RL_algorithms = ['PPO', 'TRPO', 'UATRPO', 'GePPO', 'TD3', 'SAC']
HRL = ['HPPO', 'HTRPO', 'HUATRPO', 'HGePPO', 'HTD3', 'HSAC']

colors = {}

colors['PPO'] = 'tab:blue'
colors['TRPO'] = 'tab:orange'
colors['UATRPO'] = 'tab:pink'
colors['GePPO'] = 'lime'
colors['TD3'] = 'tab:red'
colors['SAC'] = 'tab:purple'
colors['HPPO_2'] = 'tab:brown'
colors['HTRPO_2'] = 'tab:green'
colors['HUATRPO_2'] = 'tab:gray'
colors['HGePPO_2'] = 'chocolate'
colors['HTD3_2'] = 'tab:olive'
colors['HSAC_2'] = 'tab:cyan'
colors['HPPO_3'] = 'lightcoral'
colors['HTRPO_3'] = 'fuchsia'
colors['HUATRPO_3'] = 'gold'
colors['HGePPO_3'] = 'magenta'
colors['HTD3_3'] = 'lightseagreen'
colors['HSAC_3'] = 'peru'

# %% Classical Algorithms from the literature

# RL_algorithms = ['AWAC', 'AWAC_GAE', 'AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Peng', 'AWAC_Q_lambda_TB', 
#                   'GeA2C', 'GePPO', 'PPO', 'SAC', 'TD3']

RL_algorithms = ['GePPO', 'PPO', 'SAC', 'TD3']

colors = {}

Top_three = True

colors['AWAC'] = 'tab:blue'
colors['AWAC_GAE'] = 'tab:orange'
colors['AWAC_Q_lambda_Haru'] = 'lime'
colors['AWAC_Q_lambda_Peng'] = 'tab:purple'
colors['AWAC_Q_lambda_TB'] = 'tab:brown'
colors['GeA2C'] = 'tab:green'
colors['GePPO'] = 'tab:gray'
colors['PPO'] = 'chocolate'
colors['SAC'] = 'tab:olive'
colors['TD3'] = 'tab:cyan'

environments = ['MiniGrid-Empty-16x16-v0']

for env in environments:
    
    columns = 2
    rows = 1
    
    fig, ax = plt.subplots(rows, columns, figsize=(12,4))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    # fig.suptitle(env, fontsize="xx-large")
    
    for k, ax_row in enumerate([ax]):
        for j, axes in enumerate(ax_row):
    
            for i in range(len(RL_algorithms)):
    
                
                if j == 0:
                    Top_three = False
                else:
                    Top_three = True
            
                policy = RL_algorithms[i]
                
                RL = []
                
                for seed in range(10):
                    
                    try:
                        if policy == "TD3" or policy == "SAC":
                            with open(f'results_partial/RL/evaluation_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                                RL.append(np.load(f, allow_pickle=True))  
                        else:
                            with open(f'results_partial/RL/evaluation_RL_{policy}_Entropy_True_{env}_{seed}.npy', 'rb') as f:
                                RL.append(np.load(f, allow_pickle=True))    
                                
                    except:
                        continue
                        
                try:
                    if Top_three:
                        temp = []
                        RL_temp = []
                        for i in range(10):
                            temp.append([RL[i][-1], i])
                            
                        temp.sort()
                            
                        RL_temp.append(RL[temp[-1][1]])
                        RL_temp.append(RL[temp[-2][1]])
                        RL_temp.append(RL[temp[-3][1]])
                        
                        RL = RL_temp
                            
                    
                    mean = np.mean(np.array(RL), 0)
                    steps = np.linspace(0, (len(mean)-1)*4096, len(mean))
                    std = np.std(np.array(RL),0)
                    axes.plot(steps, mean, label=policy, c=colors[policy])
                    axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
                    
                    axes.set_ylim([0, 1.05])
                    axes.set_xlabel('Frames')
                    axes.set_ylabel('Reward')
                except:
                    continue

    if not os.path.exists(f"./Figures/{env}"):
        os.makedirs(f"./Figures/{env}")
   
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4, prop={'size': 12})
           
    plt.savefig(f'Figures/{env}/{env}_Online_RL.pdf', format='pdf', bbox_inches='tight')

# %% Advantage Weighted Regression

RL_algorithms = ['AWAC', 'AWAC_GAE', 'AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Peng', 'AWAC_Q_lambda_TB']

colors = {}

colors['AWAC'] = 'tab:blue'
colors['AWAC_GAE'] = 'tab:orange'
colors['AWAC_Q_lambda_Haru'] = 'lime'
colors['AWAC_Q_lambda_Peng'] = 'tab:purple'
colors['AWAC_Q_lambda_TB'] = 'tab:brown'
colors['GeA2C'] = 'tab:green'
colors['GePPO'] = 'tab:gray'
colors['PPO'] = 'chocolate'
colors['SAC'] = 'tab:olive'
colors['TD3'] = 'tab:cyan'

environments = ['MiniGrid-Empty-16x16-v0']

for env in environments:
    
    columns = 2
    rows = 1
    
    fig, ax = plt.subplots(rows, columns, figsize=(12,4))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    
    for k, ax_row in enumerate([ax]):
        for j, axes in enumerate(ax_row):
    
            for i in range(len(RL_algorithms)):
                
                if j == 0:
                    Top_three = False
                else:
                    Top_three = True
            
                policy = RL_algorithms[i]
                
                RL = []
                
                for seed in range(10):
                    
                    try:
                        if policy == "TD3" or policy == "SAC":
                            with open(f'results_partial/RL/evaluation_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                                RL.append(np.load(f, allow_pickle=True))  
                        else:
                            with open(f'results_partial/RL/evaluation_RL_{policy}_Entropy_False_{env}_{seed}.npy', 'rb') as f:
                                RL.append(np.load(f, allow_pickle=True))    
                                
                    except:
                        continue
                        
                try:
                    
                    if Top_three:
                        temp = []
                        RL_temp = []
                        for i in range(10):
                            temp.append([RL[i][-1], i])
                            
                        temp.sort()
                            
                        RL_temp.append(RL[temp[-1][1]])
                        RL_temp.append(RL[temp[-2][1]])
                        RL_temp.append(RL[temp[-3][1]])
                        
                        RL = RL_temp
                    
                    mean = np.mean(np.array(RL), 0)
                    steps = np.linspace(0, (len(mean)-1)*4096, len(mean))
                    std = np.std(np.array(RL),0)
                    axes.plot(steps, mean, label=policy, c=colors[policy])
                    axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
                    
                    axes.set_ylim([0, 1.05])
                    axes.set_xlabel('Frames')
                    axes.set_ylabel('Reward')
                except:
                    continue

    if not os.path.exists(f"./Figures/{env}"):
        os.makedirs(f"./Figures/{env}")
              
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3, prop={'size': 12})
           
    plt.savefig(f'Figures/{env}/{env}_AWR.pdf', format='pdf', bbox_inches='tight')
    
for env in environments:
    
    columns = 2
    rows = 1
    
    fig, ax = plt.subplots(rows, columns, figsize=(12,4))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    
    for k, ax_row in enumerate([ax]):
        for j, axes in enumerate(ax_row):
    
            for i in range(len(RL_algorithms)):
                
                if j == 0:
                    Top_three = False
                else:
                    Top_three = True
            
                policy = RL_algorithms[i]
                
                RL = []
                
                for seed in range(10):
                    
                    try:
                        if policy == "TD3" or policy == "SAC":
                            with open(f'results_partial/RL/evaluation_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                                RL.append(np.load(f, allow_pickle=True))  
                        else:
                            with open(f'results_partial/RL/evaluation_RL_{policy}_Entropy_True_{env}_{seed}.npy', 'rb') as f:
                                RL.append(np.load(f, allow_pickle=True))    
                                
                    except:
                        continue
                        
                try:
                    
                    if Top_three:
                        temp = []
                        RL_temp = []
                        for i in range(10):
                            temp.append([RL[i][-1], i])
                            
                        temp.sort()
                            
                        RL_temp.append(RL[temp[-1][1]])
                        RL_temp.append(RL[temp[-2][1]])
                        RL_temp.append(RL[temp[-3][1]])
                        
                        RL = RL_temp
                    
                    mean = np.mean(np.array(RL), 0)
                    steps = np.linspace(0, (len(mean)-1)*4096, len(mean))
                    std = np.std(np.array(RL),0)
                    axes.plot(steps, mean, label=policy, c=colors[policy])
                    axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
                    
                    axes.set_ylim([0, 1.05])
                    axes.set_xlabel('Frames')
                    axes.set_ylabel('Reward')
                except:
                    continue

    if not os.path.exists(f"./Figures/{env}"):
        os.makedirs(f"./Figures/{env}")
              
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3, prop={'size': 12})
           
    plt.savefig(f'Figures/{env}/{env}_AWR_Entropy_True.pdf', format='pdf', bbox_inches='tight')


# %% Offline RL from expert

RL_algorithms = ['AWAC', 'AWR', 'AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Peng', 'AWAC_Q_lambda_TB', 'BC', 'SAC_BC']

data_set = ['human_expert', 'random']
environments = ['MiniGrid-Empty-16x16-v0']

colors = {}

data_set = ['human_expert', 'random']
environments = ['MiniGrid-Empty-16x16-v0']

colors['AWAC'] = 'tab:blue'
colors['AWR'] = 'tab:orange'
colors['AWAC_Q_lambda_Haru'] = 'lime'
colors['AWAC_Q_lambda_Peng'] = 'tab:purple'
colors['AWAC_Q_lambda_TB'] = 'tab:brown'
colors['BC'] = 'tab:green'
colors['SAC_BC'] = 'tab:gray'

environments = ['MiniGrid-Empty-16x16-v0']

for env in environments:
    for data in data_set:
    
        columns = 2
        rows = 1
        
        fig, ax = plt.subplots(rows, columns, figsize=(12,4))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
        
        for k, ax_row in enumerate([ax]):
            for j, axes in enumerate(ax_row):
        
                for i in range(len(RL_algorithms)):
                    
                    if j == 0:
                        Top_three = False
                    else:
                        Top_three = True
                
                    policy = RL_algorithms[i]
                    
                    if policy == 'AWR':
                        policy_label = 'AWAC_GAE'
                    else:
                        policy_label = policy
                        
                    a_file = open(f"offline_data_set/data_set_{env}_{data}.pkl", "rb")
                    offline_set = pickle.load(a_file)   
                    
                    reward_set = np.sum(offline_set['rewards'])/np.sum(offline_set['terminals'])
                    
                    RL = []
                    
                    for seed in range(10):
                        try:
                            if policy == "BC" or policy == "SAC_BC":
                                with open(f'results_partial/offline_RL/evaluation_offline_RL_{policy}_{env}_dataset_{data}_{seed}.npy', 'rb') as f:
                                    RL.append(np.load(f, allow_pickle=True))  
                            else:
                                with open(f'results_partial/offline_RL/evaluation_offline_RL_{policy}_Entropy_False_{env}_dataset_{data}_{seed}.npy', 'rb') as f:
                                    RL.append(np.load(f, allow_pickle=True))    
                                    
                        except:
                            continue
                            
                    try:
                        
                        if Top_three:
                            temp = []
                            RL_temp = []
                            for i in range(10):
                                temp.append([RL[i][-1], i])
                                
                            temp.sort()
                                
                            RL_temp.append(RL[temp[-1][1]])
                            RL_temp.append(RL[temp[-2][1]])
                            RL_temp.append(RL[temp[-3][1]])
                            
                            RL = RL_temp
                        
                        mean = np.mean(np.array(RL), 0)
                        steps = np.linspace(0, 100, len(mean))
                        std = np.std(np.array(RL),0)
                        axes.plot(steps, mean, label=policy_label, c=colors[policy])
                        axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
                        
                        axes.set_ylim([0, 1.05])
                        axes.set_xlabel('Frames')
                        axes.set_ylabel('Reward')
                        
                        print(f'dataset: {data}, agent: {policy}, reward: {mean[-1]}, std: {std[-1]} ,Top Three: {Top_three}')
                        
                    except:
                        continue
                    
                axes.plot(steps, reward_set*np.ones((len(steps))), 'k--', label='average reward data set')
    
        if not os.path.exists(f"./Figures/{env}"):
            os.makedirs(f"./Figures/{env}")
                  
        handles, labels = axes.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3, prop={'size': 12})
               
        plt.savefig(f'Figures/{env}/offline_RL_{env}_dataset_{data}.pdf', format='pdf', bbox_inches='tight')


# %%

RL_algorithms = ['GePPO']

# RL_algorithms = ['AWAC', 'AWAC_GAE', 'AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Peng', 'AWAC_Q_lambda_TB', 
#                  'GeA2C', 'GePPO', 'PPO', 'SAC', 'TD3']

# RL_algorithms = ['PPO', 'PPO_from_videos']

colors = {}

colors['AWAC'] = 'tab:blue'
colors['AWAC_GAE'] = 'tab:orange'
colors['AWAC_Q_lambda_Haru'] = 'lime'
colors['AWAC_Q_lambda_Peng'] = 'tab:purple'
colors['AWAC_Q_lambda_TB'] = 'tab:brown'
colors['GeA2C'] = 'tab:green'
colors['GePPO'] = 'tab:gray'
colors['PPO'] = 'chocolate'
colors['SAC'] = 'tab:olive'
colors['TD3'] = 'tab:cyan'

environments = ['MiniGrid-Empty-16x16-v0']

for env in environments:
    
    columns = 1
    rows = 1
    
    fig, ax = plt.subplots(rows, columns)
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    fig.suptitle(env, fontsize="xx-large")
    
    for i in range(len(RL_algorithms)):
    
    # for k, ax_row in enumerate(ax):
    #     for j, axes in enumerate(ax_row):
            
        policy = RL_algorithms[i]
        
        RL = []
        
        for seed in range(10):
            
            try:
                if policy == "TD3" or policy == "SAC":
                    with open(f'results_partial/RL/evaluation_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                        RL.append(np.load(f, allow_pickle=True))  
                else:
                    with open(f'results_partial/RL/evaluation_RL_{policy}_Entropy_True_{env}_{seed}.npy', 'rb') as f:
                        RL.append(np.load(f, allow_pickle=True))    
                        
            except:
                continue
                
        try:
            mean = np.mean(np.array(RL), 0)
            steps = np.linspace(0, (len(mean)-1)*4096, len(mean))
            std = np.std(np.array(RL),0)
            ax.plot(steps, mean, label=policy, c=colors[policy])
            ax.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
        except:
            continue

    if not os.path.exists(f"./Figures/{env}"):
        os.makedirs(f"./Figures/{env}")
              
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    ax.set_xlabel('Frames')
    ax.set_ylabel('Reward')



# %%


for env in environments:
    
    columns = 1
    rows = 1
    
    fig, ax = plt.subplots(rows, columns)
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    fig.suptitle(env, fontsize="xx-large")
    
    for i in range(len(RL_algorithms)):
    
    # for k, ax_row in enumerate(ax):
    #     for j, axes in enumerate(ax_row):
            
        policy = RL_algorithms[i]
        
        RL = []
        wallclock = []
        
        for seed in range(10):
            
            try:
                if policy == "TD3" or policy == "SAC":
                    with open(f'results_partial/RL/evaluation_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                        RL.append(np.load(f, allow_pickle=True))  
                        
                    with open(f'results_partial/RL/wallclock_time_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                        wallclock.append(np.load(f, allow_pickle=True))  
                        
                else:
                    with open(f'results_partial/RL/evaluation_RL_{policy}_Entropy_True_{env}_{seed}.npy', 'rb') as f:
                        RL.append(np.load(f, allow_pickle=True))     
                        
                    with open(f'results_partial/RL/wallclock_time_RL_{policy}_Entropy_True_{env}_{seed}.npy', 'rb') as f:
                        wallclock.append(np.load(f, allow_pickle=True))  
                        
            except:
                continue
                
        try:
            mean = np.mean(np.array(RL), 0)
            steps = np.append(0, np.mean(np.array(wallclock)/3600, 0))
            std = np.std(np.array(RL),0)
            ax.plot(steps, mean, label=policy, c=colors[policy])
            ax.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
        except:
            continue
                
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    ax.set_xlabel('Wallclock Time [h]')
    ax.set_ylabel('Reward')
    plt.savefig(f'Figures/{env}/{env}_Online_RL_Time.pdf', format='pdf', bbox_inches='tight')

# %%
for env in environments:
    
    columns = 3
    rows = 2
    
    fig, ax = plt.subplots(rows, columns, figsize=(20,7))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    fig.suptitle(env, fontsize="xx-large")
    i = 0
    
    for k, ax_row in enumerate(ax):
        for j, axes in enumerate(ax_row):
            
            policy = RL_algorithms[i]
            
            RL = []
            HRL_2 = []
            HRL_3 = []
            
            for seed in range(10):
                try:
                    with open(f'results/HRL/evaluation_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                        RL.append(np.load(f, allow_pickle=True))  
                except:
                    print(f"{policy}_{env}_{seed} not found")
                    
                with open(f'results/HRL/evaluation_HRL_H{policy}_nOptions_2_{env}_{seed}.npy', 'rb') as f:
                    HRL_2.append(np.load(f, allow_pickle=True))   
                    
                with open(f'results/HRL/evaluation_HRL_H{policy}_nOptions_3_{env}_{seed}.npy', 'rb') as f:
                    HRL_3.append(np.load(f, allow_pickle=True)) 
                
            try:
                mean = np.mean(np.array(RL), 0)
                steps = np.linspace(0,((len(mean)-1)*specs[env]['number_steps_per_iter']),len(mean))
                std = np.std(np.array(RL),0)
                axes.plot(steps, mean, label=policy, c=colors[policy])
                axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
                
                mean = np.mean(np.array(HRL_2),0)
                std = np.std(np.array(HRL_2),0)
                axes.plot(steps, mean, label=f'H{policy} 2 options', c=colors[f'H{policy}_2'])
                axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[f'H{policy}_2'])
                
                mean = np.mean(np.array(HRL_3),0)
                std = np.std(np.array(HRL_3),0)
                axes.plot(steps, mean, label=f'H{policy} 3 options', c=colors[f'H{policy}_3'])
                axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[f'H{policy}_3'])
                
            except:
                RL_new = []
                for s in range(len(RL)):
                    RL_new.append(RL[s][0:301])
                    
                mean = np.mean(np.array(RL_new), 0)
                steps = np.linspace(0,((len(mean)-1)*specs[env]['number_steps_per_iter']),len(mean))
                std = np.std(np.array(RL_new),0)
                axes.plot(steps, mean, label=policy, c=colors[policy])
                axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
                
                mean = np.mean(np.array(HRL_2),0)
                std = np.std(np.array(HRL_2),0)
                axes.plot(steps, mean, label=f'H{policy} 2 options', c=colors[f'H{policy}_2'])
                axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[f'H{policy}_2'])
                
                mean = np.mean(np.array(HRL_3),0)
                std = np.std(np.array(HRL_3),0)
                axes.plot(steps, mean, label=f'H{policy} 3 options', c=colors[f'H{policy}_3'])
                axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[f'H{policy}_3'])
            
            axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
            axes.set_xlabel('Steps')
            axes.set_ylabel('Reward')
            
            i+=1
            
    if not os.path.exists(f"./Figures/{env}"):
        os.makedirs(f"./Figures/{env}")
    
    plt.savefig(f'Figures/{env}/{env}_comparison.pdf', format='pdf', bbox_inches='tight')
            
            
            
            
    
    
