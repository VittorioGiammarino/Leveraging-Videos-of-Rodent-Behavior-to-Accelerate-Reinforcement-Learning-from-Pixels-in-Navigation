# Leveraging-Videos-of-Rodent-Behavior-to-Accelerate-Reinforcement-Learning-from-Pixels-in-Navigation

To install the anaconda environment follow the instruction here 
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
the environment.yml file contains more packages than those actually needed. 

## TORUN

We care only about the Minigrid folder: if you run on the SCC the bash files parallelize_RL.sh and parallelize_on_off_RL_from_observations_rodent_ri_***.sh will send 10 batch submission to the cluster
each batch file is associated to a QSUB file which is in the QSUB folder.

Watch out that the .qsub file calls the miniconda environment, you should change the line with the folder in which you save the environment on the SCC 
