#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py train| tee runninglogs/save/train_hg_exp2.txt