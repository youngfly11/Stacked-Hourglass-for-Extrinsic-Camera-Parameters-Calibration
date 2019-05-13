#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train | tee runninglogs/save/train_exp3.txt