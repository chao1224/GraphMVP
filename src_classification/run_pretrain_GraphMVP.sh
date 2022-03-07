#!/usr/bin/env bash

source $HOME/.bashrc
conda activate PC3D
conda deactivate
conda activate PC3D


echo $@
date
echo "start"
python pretrain_GraphMVP.py $@
echo "end"
date
