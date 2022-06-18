#!/bin/bash

for seed in $(seq 0 2);
do
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_domain_adaptation_ri_0.01/AWAC_GAE.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_domain_adaptation_ri_0.01/AWAC_Q_lambda_Haru.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_domain_adaptation_ri_0.01/AWAC_Q_lambda_Peng.qsub $seed 
#qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_domain_adaptation_ri_0.01/AWAC_Q_lambda_TB.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_domain_adaptation_ri_0.01/SAC.qsub $seed 
qsub -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_domain_adaptation_ri_0.01/AWAC.qsub $seed 

done 
