#!/bin/bash

CUDA_VISIBLE_DEVICES=6 nohup python3 -u train_student.py --path_t ./res/wrn_40_2/ckpt-233.t7 --distill mkd --model_s ShuffleV1 -r 0.5 -a 0.5 -b 0 --kd_reduction mean --kd_T 50 --trial 1 > mkd_wrn_mean.txt & 
CUDA_VISIBLE_DEVICES=6 nohup python3 -u train_student.py --path_t ./res/resnet32x4/ckpt-220.t7 --distill mkd --model_s ShuffleV1 -r 0.5 -a 0.5 -b 0 --kd_reduction mean --kd_T 50 --trial 1 > mkd_resnet32x4_shufflenet_mean.txt &
CUDA_VISIBLE_DEVICES=6 nohup python3 -u train_student.py --path_t ./res/resnet32x4/ckpt-220.t7 --distill mkd --model_s ShuffleV2 -r 0.5 -a 0.5 -b 0 --kd_reduction mean --kd_T 50 --trial 1 > mkd_resnet32x4_shufflenetv2_mean.txt 
CUDA_VISIBLE_DEVICES=6 nohup python3 -u train_student.py --path_t ./res/resnet50/ckpt-229.t7 --distill mkd --model_s MobileNetV2 -r 0.5 -a 0.5 -b 0 --kd_reduction mean --kd_T 50 --trial 1 > mkd_resnet50_mobilenetv2_mean.txt &
CUDA_VISIBLE_DEVICES=6 nohup python3 -u train_student.py --path_t ./res/resnet50/ckpt-229.t7 --distill mkd --model_s vgg8 -r 0.5 -a 0.5 -b 0 --kd_reduction mean --kd_T 50 --trial 2 > mkd_resnet50_vgg8_mean_2.txt &
CUDA_VISIBLE_DEVICES=6 nohup python3 -u train_student.py --path_t ./res/vgg13/ckpt-232.t7 --distill mkd --model_s MobileNetV2 -r 0.5 -a 0.5 -b 0 --kd_reduction mean --kd_T 50 --trial 1 > mkd_vgg13_mobilenetv2_mean.txt