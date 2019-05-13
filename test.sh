#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py test | tee runninglogs/save/exp1/test.txt