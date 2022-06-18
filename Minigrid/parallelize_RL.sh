#!/bin/bash

for seed in $(seq 0 9);
do
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G SAC.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G TD3.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G PPO.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G Vanilla_A2C.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G A2C.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G GePPO.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16G GeA2C.qsub $seed 

done 
