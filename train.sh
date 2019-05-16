#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,3 python -u main.py train | tee runninglogs/save/train_hg_exp3.txt