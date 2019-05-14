#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,3 python -u main.py test | tee runninglogs/save/test-hg-exp4.txt