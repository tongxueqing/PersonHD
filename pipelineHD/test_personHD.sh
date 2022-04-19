#!/bin/bash

python scripts/test_pose_transfer_model.py \
    --id  personHD_finetune_jitter_2e5_front \
    --gpu_ids 7 \
    --dataset_name personHD_2e5_front \
    --which_model_G dual_unet \
    --G_feat_warp 1 \
    --G_vis_mode residual \
    --pretrained_flow_id FlowReg \
    --pretrained_flow_epoch best \
    --dataset_type pose_transfer_parsing_personHD \
    --which_epoch 40 \
    --batch_size 4 \
    --save_output \
    --output_dir output_40



