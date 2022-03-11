#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:01:40 2022

@author: vittoriogiammarino
"""
import os
import numpy as np
import matplotlib.pyplot as plt

with open('data_set/rodent_frames_data_set.npy', 'rb') as f:
    data_set = np.load(f, allow_pickle=True)
    
# %% processed data _set

episodes_states = []
intrinsic_reward = 0.001
t = 0

for i in range(0, len(data_set), 25):
    episodes_states.append(data_set[i].transpose(2,0,1))
    
Data_set = np.array(episodes_states)

# %%

np.save("data_set/rodent_data_processed.npy", Data_set)

# %%

if not os.path.exists("./data_set/Figures"):
    os.makedirs("./data_set/Figures")


columns = 10
rows = 2
fig, ax = plt.subplots(rows, columns, figsize=(15,3.2))

i = 2010

for k, ax_row in enumerate(ax):
    for j, axes in enumerate(ax_row):
        
        axes.imshow(Data_set[i].transpose(1,2,0))
        axes.set_axis_off()
        i+=1

fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f'data_set/Figures/frames_{i}.pdf', format='pdf', bbox_inches='tight')

# %%

img = env.reset()

# %%

columns = 2
rows = 1
fig, ax = plt.subplots(rows, columns, figsize=(5,3.2))

i = 2010

for k, ax_row in enumerate([ax]):
    for j, axes in enumerate(ax_row):
        
        if j==0:
            axes.imshow(img)
        else:
            axes.imshow(Data_set[i].transpose(1,2,0))
        
        axes.set_axis_off()
        i+=1

fig.subplots_adjust(wspace=0.03, hspace=0)
plt.savefig(f'data_set/Figures/isomorphic_envs.pdf', format='pdf', bbox_inches='tight')

