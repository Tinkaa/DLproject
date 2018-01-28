#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem-per-cpu 200G
#SBATCH -t 30:00:00

module load anaconda3
source activate /scratch/work/phama1/tensorflow

srun --gres=gpu:1 python train_frcnn.py -p VOCdevkit --config_filename config_focal_loss.pickle --output_weight_path ./model_focal_loss.hdf5 --num_epochs 120
