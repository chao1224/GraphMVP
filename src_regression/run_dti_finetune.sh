#!/usr/bin/env bash

source $HOME/.bashrc
conda activate graphMVP
conda deactivate
conda activate graphMVP

echo $@
date

echo "start"
python dti_finetune.py $@
echo "end"
date
