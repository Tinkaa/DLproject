#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem-per-cpu 200G
#SBATCH -t 4:00:00

module load anaconda3
source activate /scratch/work/phama1/tensorflow

srun --gres=gpu:1 python frcnn_subm_VOC2007.py -p VOCdevkit
