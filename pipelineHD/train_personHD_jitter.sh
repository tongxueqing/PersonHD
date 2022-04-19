#!/bin/bash

python scripts/train_pose_transfer_model.py \
    --id personHD_2e5_front_jitter \
    --gpu_ids 0,1,2,3,5,7 \
    --dataset_name personHD_2e5_front \
    --which_model_G dual_unet \
    --G_feat_warp 1 \
    --G_vis_mode residual \
    --pretrained_flow_id FlowReg \
    --pretrained_flow_epoch best \
    --dataset_type pose_transfer_parsing_personHD_jitter \
    --check_grad_freq 3000 \
    --batch_size 12 \
    --n_epoch 10 \

