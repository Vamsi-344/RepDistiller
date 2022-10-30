#!/bin/bash

CUDA_VISIBLE_DEVICES=5 nohup python3 -u train_student.py --path_t ./res/wrn_40_2/ckpt-233.t7 --distill cmcrd --model_s ShuffleV1 -a 0 -b 0.8 --kd_T 50 --trial 1 > cmcrd_wrn.txt & 
CUDA_VISIBLE_DEVICES=5 nohup python3 -u train_student.py --path_t ./res/resnet32x4/ckpt-220.t7 --distill cmcrd --model_s ShuffleV1 -a 0 -b 0.8 --kd_T 50 --trial 1 > cmcrd_resnet32x4_shufflenet.txt &
CUDA_VISIBLE_DEVICES=5 nohup python3 -u train_student.py --path_t ./res/resnet32x4/ckpt-220.t7 --distill cmcrd --model_s ShuffleV2 -a 0 -b 0.8 --kd_T 50 --trial 1 > cmcrd_resnet32x4_shufflenetv2.txt 
CUDA_VISIBLE_DEVICES=5 nohup python3 -u train_student.py --path_t ./res/resnet50/ckpt-229.t7 --distill cmcrd --model_s MobileNetV2 -a 0 -b 0.8 --kd_T 50 --trial 1 > cmcrd_resnet50_mobilenetv2.txt &
CUDA_VISIBLE_DEVICES=5 nohup python3 -u train_student.py --path_t ./res/resnet50/ckpt-229.t7 --distill cmcrd --model_s vgg8 -a 0 -b 0.8 --kd_T 50 --trial 1 > cmcrd_resnet50_vgg8.txt &
CUDA_VISIBLE_DEVICES=5 nohup python3 -u train_student.py --path_t ./res/vgg13/ckpt-232.t7 --distill cmcrd --model_s MobileNetV2 -a 0 -b 0.8 --kd_T 50 --trial 1 > cmcrd_vgg13_mobilenetv2.txt