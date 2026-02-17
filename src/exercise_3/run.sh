#!/bin/bash

# Read partition
gpu=$1

# Submit job to queue
sbatch --partition=$gpu                                            \
       --nodes=1                                                   \
       --mem=30G                                                   \
       --ntasks-per-node=1                                         \
       --gres=gpu:1                                                \
       --cpus-per-task=4                                           \
       --time=1-00:00:00                                           \
       --job-name="U-Net"                                          \
       --output=out.dat                                            \
       --error=err.dat                                             \
       --mail-type=ALL,TIME_LIMIT_50,TIME_LIMIT_90,TIME_LIMIT      \
       --mail-user=$USER@uni-muenster.de                           \
       --wrap "ml purge;                                           \
               ml palma/2022a;                                     \
               ml GCCcore/11.3.0;                                  \
               ml GCC/11.3.0;                                      \
               ml OpenMPI/4.1.4;                                   \
               ml Python/3.10.4;                                   \
               ml torchvision/0.13.1;                              \
               srun python solution.py"