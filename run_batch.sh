#!/bin/bash

CUDA_VISIBLE_DEVICES=2 nohup python3 -u train_student.py --path_t ./res/wrn_40_2/ckpt-233.t7 --distill crd --model_s ShuffleV1 -a 0 -b 0.8 --kd_T 50 --trial 1 > crd.txt & 
CUDA_VISIBLE_DEVICES=3 nohup python3 -u train_student.py --path_t ./res/wrn_40_2/ckpt-233.t7 --distill kd --model_s ShuffleV1 -a 0 -b 0.8 --kd_T 50 --trial 1 > kd.txt