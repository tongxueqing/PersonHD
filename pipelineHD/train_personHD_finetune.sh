#!/bin/bash

python scripts/train_pose_transfer_model.py \
    --id personHD_finetune_jitter_2e5_front \
    --gpu_ids 2,3 \
    --dataset_name personHD_finetune_2e5_front \
    --which_model_G dual_unet \
    --G_feat_warp 1 \
    --G_vis_mode residual \
    --pretrained_flow_id FlowReg \
    --pretrained_flow_epoch best \
    --dataset_type pose_transfer_parsing_personHD \
    --check_grad_freq 3000 \
    --batch_size 4 \
    --n_epoch 100 \
    --pretrained_G_id PoseTransfer_personHD_2e5_front_jitter \
    --pretrained_G_epoch 8 \
    --save_epoch_freq 10
    