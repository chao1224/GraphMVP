#!/usr/bin/env bash

source $HOME/.bashrc
conda activate GraphMVP

echo $@
date
echo "start"
python pretrain_GraphMVP_hybrid.py $@
echo "end"
date
