#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py test | tee runninglogs/save/test-hg-exp3.txt