#!/bin/bash

for seed in $(seq 0 2);
do
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G QSUBS/Discrete/RL/AWAC.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G QSUBS/Discrete/RL/AWAC_GAE.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G QSUBS/Discrete/RL/AWAC_Q_lambda_Haru.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G QSUBS/Discrete/RL/AWAC_Q_lambda_Peng.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G QSUBS/Discrete/RL/AWAC_Q_lambda_TB.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G QSUBS/Discrete/RL/GePPO.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G QSUBS/Discrete/RL/PPO.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G QSUBS/Discrete/RL/SAC.qsub $seed
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G QSUBS/Discrete/RL/TD3.qsub $seed  

done 
