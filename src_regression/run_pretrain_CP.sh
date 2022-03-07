#!/usr/bin/env bash

source $HOME/.bashrc
conda activate PC3D
conda deactivate
conda activate PC3D

echo $@
date
echo "start"
python pretrain_CP.py $@
echo "end"
date
