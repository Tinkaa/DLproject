#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH -p gpu
#SBATCH --mem-per-cpu 100G
#SBATCH -t 20:00:00

module load anaconda3 CUDA/7.5.18 cudnn/5
source activate /scratch/work/phama1/tensorflow

srun --gres=gpu:4 python train_frcnn.py -p VOCdevkit
