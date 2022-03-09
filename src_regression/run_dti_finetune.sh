#!/usr/bin/env bash

source $HOME/.bashrc
conda activate GraphMVP
conda deactivate
conda activate GraphMVP

echo $@
date

echo "start"
python dti_finetune.py $@
echo "end"
date
