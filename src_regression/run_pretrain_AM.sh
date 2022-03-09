#!/usr/bin/env bash

source $HOME/.bashrc
conda activate GraphMVP
conda deactivate
conda activate GraphMVP

echo $@
date
echo "start"
python pretrain_AM.py $@
echo "end"
date
