#!/bin/bash

for seed in $(seq 0 9);
do
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 SAC.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 TD3.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/RL/PPO.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 Vanilla_A2C.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 A2C.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 GePPO.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 GeA2C.qsub $seed 

done 
