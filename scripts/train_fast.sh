#!/bin/bash

cd ../tools

python train_net.py \
    --dataset_name          annots \
    --json_annotation_train annots/train.json \
    --image_path_train      annots/images \
    --json_annotation_val   annots/test.json \
    --image_path_val        annots/images \
    --config-file           ../configs/prima/fast_rcnn_R_50_FPN_3x.yaml \
    OUTPUT_DIR  ../outputs/prima/mask_rcnn_R_50_FPN_3x/ \
    SOLVER.IMS_PER_BATCH 2 
