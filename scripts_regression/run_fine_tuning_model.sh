#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --time=2:59:00
#SBATCH --ntasks=1
#SBATCH --array=0-2%3
#SBATCH --output=logs/%j.out
#SBATCH --job-name=reg


###############SBATCH --gres=gpu:v100l:1

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
export dataset_list=(esol freesolv lipophilicity malaria cep)
export split_list=(scaffold random)
export seed_list=(0 1 2 3 4 5 6 7 8 9)
export batch_size=256
export mode=$1
export seed=${seed_list[$SLURM_ARRAY_TASK_ID]}





if [ "$mode" == "random" ]; then

    for dataset in "${dataset_list[@]}"; do
    for split in "${split_list[@]}"; do
        export folder="$mode"/"$seed"/"$split"
        mkdir -p ./output/"$folder"/"$dataset"

        export output_path=./output/"$folder"/"$dataset".out
        export output_model_dir=./output/"$folder"/"$dataset"

        echo "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID" > "$output_path"
        echo `date` >> "$output_path"

        bash ./run_molecule_finetune_regression.sh \
        --dataset="$dataset" --runseed="$seed" --eval_train --batch_size="$batch_size" \
        --dropout_ratio=0.2 --split="$split" \
        --output_model_dir="$output_model_dir" \
        >> "$output_path"

        echo `date` >> "$output_path"
    done
    done




else

    for dataset in "${dataset_list[@]}"; do
    for split in "${split_list[@]}"; do
        export folder="$mode"/"$seed"/"$split"
        mkdir -p ./output/"$folder"
        mkdir -p ./output/"$folder"/"$dataset"

        export output_path=./output/"$folder"/"$dataset".out
        export output_model_dir=./output/"$folder"/"$dataset"
        export input_model_file=./output/"$mode"/pretraining_model.pth

        echo "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID" > "$output_path"
        echo `date` >> "$output_path"

        bash ./run_molecule_finetune_regression.sh \
        --dataset="$dataset" --runseed="$seed" --eval_train --batch_size="$batch_size" \
        --dropout_ratio=0.2 --split="$split" \
        --input_model_file="$input_model_file" \
        >> "$output_path"

        echo `date` >> "$output_path"
    done
    done

fi

