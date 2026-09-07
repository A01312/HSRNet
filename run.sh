#!/bin/bash
root_dir=./data
tau=0.015 
margin=0.1
select_ratio=0.3
loss=TAL
DATASET_NAME=ICFG-PEDES 
# CUHK-PEDES ICFG-PEDES RSTPReid

CUDA_VISIBLE_DEVICES=2 \
    python train.py \
    --name HSRNet \
    --img_aug \
    --txt_aug \
    --batch_size 128 \
    --select_ratio $select_ratio \
    --tau $tau \
    --root_dir $root_dir \
    --output_dir run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --loss_names ${loss}+sr${select_ratio}_tau${tau}_margin${margin}  \
    --num_epoch 60 \
    --use_occlusion \
    --enable_parsing \
    --parsing_model_path /home/a6027/data/exp-schp-201908301523-atr.pth \
    --parsing_fusion_dim 256 \
    --parsing_weight 0.1 \
    --region_conf_thres 0.5 \
    --edge_conf_thres 0.5 \
    --occ_rate_thres 0.5 \
    --lambda_reconstruct 1.0
 