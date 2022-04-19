python scripts/test_pose_transfer_model.py \
    --id personHD_2e5_front_512 \
    --gpu_ids 6 \
    --dataset_name personHD_2e5_front_512 \
    --which_model_G dual_unet \
    --G_feat_warp 1 \
    --G_vis_mode residual \
    --pretrained_flow_id FlowReg_deepfashion \
    --pretrained_flow_epoch best \
    --dataset_type pose_transfer_parsing_personHD_512 \
    --which_epoch $i \
    --batch_size 4 \
    --save_output \
    --output_dir output_$i

python mmsr/test.py -opt "oupsampling_module_options/test_unresize_default/test_C2_matching_personHD_512_side_jitter.yml"