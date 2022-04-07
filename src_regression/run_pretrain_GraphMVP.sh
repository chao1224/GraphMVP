#!/usr/bin/env bash

source $HOME/.bashrc
conda activate GraphMVP

echo $@
date
echo "start"
python pretrain_GraphMVP.py $@
echo "end"
date
