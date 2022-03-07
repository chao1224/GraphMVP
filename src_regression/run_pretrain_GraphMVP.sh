#!/usr/bin/env bash

source $HOME/.bashrc
conda activate graphMVP
conda deactivate
conda activate graphMVP

echo $@
date
echo "start"
python pretrain_GraphMVP.py $@
echo "end"
date
