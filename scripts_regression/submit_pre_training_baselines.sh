#!/usr/bin/env bash

cd ../src_regression

epochs=100
time=2
mode_list=(AM CP GraphCL JOAO JOAOv2)
dropout_ratio=0
dataset=GEOM_2D_nmol50000_nconf1_nupper1000_morefeat


for mode in "${mode_list[@]}"; do
     export folder="$mode"/"$dataset"/epochs_"$epochs"_"$dropout_ratio"
     echo "$folder"

     mkdir -p ./output/"$folder"
     ls ./output/"$folder"

     export output_file=./output/"$folder"/pretraining.out
     export output_model_dir=./output/"$folder"/pretraining
     
     sbatch --gres=gpu:v100l:1 -c 8 --mem=32G -t "$time":59:00  --account=rrg-bengioy-ad --qos=high --job-name=baselines \
     --output="$output_file" \
     ./run_pretrain_"$mode".sh \
     --epochs="$epochs" \
     --dataset="$dataset" \
     --batch_size=256 \
     --dropout_ratio="$dropout_ratio" --num_workers=8 \
     --output_model_dir="$output_model_dir"
done
