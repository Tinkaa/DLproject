#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -p gpushort
#SBATCH --mem-per-cpu 200G
#SBATCH -t 4:00:00

module load anaconda3
source activate /scratch/work/phama1/tensorflow

srun --gres=gpu:1 python train_frcnn.py -p VOCdevkit \
  --config_filename config_all_classes.pickle \
  --output_weight_path ./model_all_classes_2.hdf5 \
  --num_epochs 18 \
  --input_weight_path model_all_classes.hdf5 
