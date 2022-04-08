#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --time=5:55:00
#SBATCH --ntasks=1
#SBATCH --array=0-2%3
#SBATCH --output=logs/%j.out
#SBATCH --job-name=repurpose_complete


###############SBATCH --gres=gpu:v100l:1

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
export dataset_list=(davis kiba)
export seed_list=(0 1 2 3 4 5 6 7 8 9)
export batch_size=256
export mode=$1
export seed=${seed_list[$SLURM_ARRAY_TASK_ID]}





if [ "$mode" == "random" ]; then

    for dataset in "${dataset_list[@]}"; do
        export folder="$mode"/"$seed"
        mkdir -p ./output/"$folder"
        mkdir -p ./output/"$folder"/"$dataset"

        export output_path=./output/"$folder"/"$dataset".out
        export output_model_dir=./output/"$folder"/"$dataset"

        echo "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID" > "$output_path"
        echo `date` >> "$output_path"

        bash ./run_dti_finetune.sh \
        --dataset="$dataset" --runseed="$seed" --batch_size="$batch_size" \
        >> "$output_path"

        echo `date` >> "$output_path"
    done




else

    for dataset in "${dataset_list[@]}"; do
        export folder="$mode"/"$seed"
        mkdir -p ./output/"$folder"
        mkdir -p ./output/"$folder"/"$dataset"

        export output_path=./output/"$folder"/"$dataset".out
        export output_model_dir=./output/"$folder"/"$dataset"
        export input_model_file=./output/"$mode"/pretraining_model.pth

        echo "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID" > "$output_path"
        echo `date` >> "$output_path"

        bash ./run_dti_finetune.sh \
        --dataset="$dataset" --runseed="$seed" --batch_size="$batch_size" \
        --input_model_file="$input_model_file" \
        >> "$output_path"

        echo `date` >> "$output_path"
    done

fi

