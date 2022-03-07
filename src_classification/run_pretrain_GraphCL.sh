#!/usr/bin/env bash

source $HOME/.bashrc
conda activate GraphMVP
conda deactivate
conda activate GraphMVP

echo $@
date

echo "start"
python pretrain_GraphCL.py $@
echo "end"
date
