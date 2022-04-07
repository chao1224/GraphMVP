#!/usr/bin/env bash

source $HOME/.bashrc
conda activate GraphMVP

echo $@
date
echo "start"
python pretrain_CP.py $@
echo "end"
date
