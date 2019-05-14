#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019-04-28 10:12
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import logging
import time
from distutils.version import LooseVersion
from sacred import Experiment
from easydict import EasyDict as edict
from lib.utils import random_init, create_logger, check_para_correctness
# from lib.classifier import Regressor
from lib.classifier_hg import Regressor

ex = Experiment()

@ex.command
def train(_run, _rnd, _seed):
    cfg = edict(_run.config)
    random_init(cfg.seed)
    check_para_correctness(cfg)
    ex.logger = create_logger(cfg, postfix='_train')
    regressor = Regressor(cfg, ex.logger)
    regressor.train()


@ex.command
def test(_run, _rnd, _seed):
    cfg = edict(_run.config)
    random_init(cfg.seed)
    check_para_correctness(cfg)
    ex.logger = create_logger(cfg, postfix='_test')
    regressor = Regressor(cfg, ex.logger)
    regressor.test_only()

if __name__ == '__main__':
    # assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), \
    #     'PyTorch>=1.0.0 is required'

    ex.add_config('./experiments/c-exp-2.yaml')
    ex.run_commandline()
