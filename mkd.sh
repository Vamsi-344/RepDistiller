#!/bin/bash

CUDA_VISIBLE_DEVICES=3 nohup python3 -u train_student.py --path_t ./res/resnet50/ckpt-229.t7 --distill mkd --model_s vgg8 -r 0.5 -a 0.5 -b 0 --kd_reduction mean --kd_T 50 --trial 1 > mkd_resnet50_vgg8_mean.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 -u train_student.py --path_t ./res/resnet50/ckpt-229.t7 --distill mkd --model_s vgg8 -r 0.5 -a 0.5 -b 0 --kd_reduction batchmean --kd_T 50 --trial 1 > mkd_resnet50_vgg8_batchmean.txt