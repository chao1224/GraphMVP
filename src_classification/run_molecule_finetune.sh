#!/usr/bin/env bash

source $HOME/.bashrc
conda activate GraphMVP

echo $@
date

echo "start"
python molecule_finetune.py $@
echo "end"
date
