#!/usr/bin/env bash

source $HOME/.bashrc
conda activate PC3D
conda deactivate
conda activate PC3D

#cp -r ../datasets/GEOM_* $SLURM_TMPDIR
#
#echo $@
#date
#echo "start"
#python pretrain_3D_hybrid_02.py --input_data_dir="$SLURM_TMPDIR" $@
#echo "end"
#date


echo $@
date
echo "start"
python pretrain_3D_hybrid_02.py $@
echo "end"
date
