#!/usr/bin/env bash

source $HOME/.bashrc
conda activate GraphMVP

echo $@
date

echo "start"
python pretrain_JOAOv2.py $@
echo "end"
date
