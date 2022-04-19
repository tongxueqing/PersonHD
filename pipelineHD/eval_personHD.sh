#!/bin/bash

python tools/metrics_personHD.py \
--gt_path /data2/xueqing_tong/dataset/cropped_side/resize512/image_cropped_side/GT_side_1e4 \
--distorated_path checkpoints/PoseTransfer_personHD_2e5_side_512/output \
--fid_real_path /data2/xueqing_tong/dataset/cropped_side/resize512/image_cropped_side/train \
--seg_path /data2/xueqing_tong/dataset/cropped_side/resize512/mask_cropped_side/GT_side_1e4_mask >> 2e5_side_512.txt

python tools/metrics_SSIM.py \
--gt_path /data2/xueqing_tong/dataset/cropped_side/resize512/image_cropped_side/GT_side_1e4 \
--distorated_path checkpoints/PoseTransfer_personHD_2e5_side_512/output \

python tools/metrics_personHD.py \
--gt_path /data2/xueqing_tong/dataset/cropped_front/resize512/GT_front_1e4 \
--distorated_path checkpoints/PoseTransfer_personHD_2e5_front_512/output \
--fid_real_path /data2/xueqing_tong/dataset/cropped_front/resize512/train \
--seg_path /data2/xueqing_tong/dataset/cropped_front/resize512/GT_front_1e4_mask >> 2e5_front_512.txt

python tools/metrics_SSIM.py \
--gt_path /data2/xueqing_tong/dataset/cropped_front/resize512/GT_front_1e4 \
--distorated_path checkpoints/PoseTransfer_personHD_2e5_front_512/output \


