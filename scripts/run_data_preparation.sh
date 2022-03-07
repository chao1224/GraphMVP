#!/bin/bash

#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --array=0
#SBATCH --output=output/%j.out


cd $HOME
source $HOME/.bashrc
conda activate PC3D
conda deactivate
conda activate PC3D

cd ~/scratch/3D_SSL/src

# pretrain_3D_02
# 24h
#
# pretrain_3D_03
# 3h
#
# GEOM_01
# 6h

output_file=../output/GEOM_02_dataset_preparation_GEOM_04_3D_New.out

echo `date` > "$output_file"
python dataset_preparation_04.py >> "$output_file"
echo `date` >> "$output_file"
