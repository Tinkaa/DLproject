#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem-per-cpu 200G
#SBATCH -t 20:00:00

module load anaconda3
source activate /scratch/work/phama1/tensorflow

srun --gres=gpu:1 python train_frcnn.py -p VOCdevkit \
  --config_filename config_focal_loss.pickle \
  --output_weight_path ./model_focal_loss_4.hdf5 \
  --num_epochs 42 \
  --loss_func focal_loss \
  --input_weight_path model_focal_loss_3.hdf5

