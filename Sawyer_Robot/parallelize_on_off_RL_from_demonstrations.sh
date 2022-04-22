#!/bin/bash

for seed in $(seq 0 9);
do
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_0.1/AWAC_GAE.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_0.1/AWAC_Q_lambda_Haru.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_0.1/AWAC_Q_lambda_Peng.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_0.1/AWAC_Q_lambda_TB.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_0.1/SAC.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_0.1/AWAC.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_1/AWAC_GAE.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_1/AWAC_Q_lambda_Haru.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_1/AWAC_Q_lambda_Peng.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_1/AWAC_Q_lambda_TB.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_0.1/SAC.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/Discrete/on_off_RL_demonstrations/ri_0.1/AWAC.qsub $seed 

done 
