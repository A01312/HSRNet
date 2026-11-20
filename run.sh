#!/bin/bash
root_dir=${DATA_ROOT:-./data}
tau=0.015 
margin=0.1
select_ratio=0.3
loss=${LOSS_NAMES:-tri+id+re}
DATASET_NAME=${DATASET_NAME:-ICFG-PEDES} 
# CUHK-PEDES ICFG-PEDES RSTPReid

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    python train.py \
    --name HSRNet \
    --img_aug \
    --batch_size 64 \
    --select_ratio $select_ratio \
    --tau $tau \
    --root_dir "$root_dir" \
    --output_dir run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --loss_names "$loss" \
    --num_epoch 60 \
    --use_occlusion \
    --enable_parsing \
    --parsing_model_path "${SCHP_WEIGHTS:-exp-schp-201908301523-atr.pth}" \
    --parsing_fusion_dim 256 \
    --parsing_weight 0.1 \
    --region_conf_thres 0.5 \
    --edge_conf_thres 0.5 \
    --body_conf_thres 0.5 "$@"
