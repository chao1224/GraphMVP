#!/usr/bin/env bash

cd ../src_classification


export mode=GraphMVP_hybrid
export dataset_list=(GEOM_3D_nmol50000_nconf5_nupper1000)
export epochs=100





# For SchNet and GNN
export schnet_lr_scale_list=(0.1)
export num_interactions=6
export num_gaussians=51
export cutoff=10
export dropout_ratio_list=(0)
export SSL_masking_ratio_list=(0.15 0.3)




# For CL
# export CL_similarity_metric_list=(InfoNCE_dot_prod EBM_dot_prod)
export CL_similarity_metric_list=(EBM_dot_prod)
# export T_list=(0.05 0.1 0.2 0.5 1 2)
export T_list=(0.1 0.2)
export normalize_list=(normalize)



# For VAE
export AE_model=VAE
# export AE_loss_list=(l1 l2 cosine)
export AE_loss_list=(l2)
# export detach_list=(detach_target no_detach_target)
export detach_list=(detach_target)
# export beta_list=(0.1 1 2)
export beta_list=(1 2)





# For CL + VAE
export alpha_1_list=(1)
export alpha_2_list=(0.1 1)
export alpha_3_list=(0.1 1)
export SSL_2D_mode_list=(     CP    AM   IG  JOAOv2   JOAO   GraphCL)
export time_list=(            3     3    3   6        6      3)
export time_list=(            9     9    6   12       12     6)




for dataset in "${dataset_list[@]}"; do
for SSL_masking_ratio in "${SSL_masking_ratio_list[@]}"; do

for i in {0..1}; do
SSL_2D_mode=${SSL_2D_mode_list[$i]}
time=${time_list[$i]}

for alpha_3 in "${alpha_3_list[@]}"; do

for alpha_1 in "${alpha_1_list[@]}"; do
for alpha_2 in "${alpha_2_list[@]}"; do
for CL_similarity_metric in "${CL_similarity_metric_list[@]}"; do
for normalize in "${normalize_list[@]}"; do
for T in "${T_list[@]}"; do
for AE_loss in "${AE_loss_list[@]}"; do
for detach in "${detach_list[@]}"; do
for beta in "${beta_list[@]}"; do


for schnet_lr_scale in "${schnet_lr_scale_list[@]}"; do
for dropout_ratio in "${dropout_ratio_list[@]}"; do
     export folder="$mode"/"$dataset"/CL_"$alpha_1"_"$AE_model"_"$alpha_2"_"$SSL_2D_mode"_"$alpha_3"/"$num_interactions"_"$num_gaussians"_"$cutoff"_"$schnet_lr_scale"/"$SSL_masking_ratio"_"$CL_similarity_metric"_"$T"_"$normalize"_"$AE_loss"_"$detach"_"$beta"_"$epochs"_"$dropout_ratio"
     
     echo "$folder"
     mkdir -p ../output/"$folder"
     ls ../output/"$folder"

     export output_file=../output/"$folder"/pretraining.out
     export output_model_dir=../output/"$folder"/pretraining

     echo "$output_model_dir"_model_final.pth undone
     ls "$output_model_dir"*
     echo "$output_file"
     ls ../output/"$folder"
     rm ../output/"$folder"/*


     sbatch --gres=gpu:v100l:1 -c 8 --mem=32G -t "$time":00:00  --account=rrg-bengioy-ad --qos=high --job-name=CL_VAE_"$SSL_2D_mode"_"$time" \
     --output="$output_file" \
     ./run_pretrain_"$mode".sh \
     --epochs="$epochs" \
     --dataset="$dataset" \
     --batch_size=256 \
     --SSL_masking_ratio="$SSL_masking_ratio" \
     --CL_similarity_metric="$CL_similarity_metric" --T="$T" --"$normalize" \
     --AE_model="$AE_model" --AE_loss="$AE_loss" --"$detach" --beta="$beta" \
     --alpha_1="$alpha_1" --alpha_2="$alpha_2" \
     --SSL_2D_mode="$SSL_2D_mode" --alpha_3="$alpha_3" \
     --num_interactions="$num_interactions" --num_gaussians="$num_gaussians" --cutoff="$cutoff" --schnet_lr_scale="$schnet_lr_scale" \
     --dropout_ratio="$dropout_ratio" --num_workers=8 \
     --output_model_dir="$output_model_dir"

     echo

done
done

done
done
done
done
done
done
done
done
done

done
done
done
