#!/usr/bin/env bash

source $HOME/.bashrc
conda activate graphMVP
conda deactivate
conda activate graphMVP

echo $@
date
echo "start"
python pretrain_GraphMVP_hybrid.py $@
echo "end"
date
