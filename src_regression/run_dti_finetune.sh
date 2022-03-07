#!/usr/bin/env bash

source $HOME/.bashrc
conda activate PC3D
conda deactivate
conda activate PC3D

echo $@
date

echo "start"
python dti_finetune.py $@
echo "end"
date
