import time
import os
import logging
from logging.config import dictConfig
import torch
import random
import numpy as np
import os.path as osp

logging_config = dict(
    version = 1,
    formatters = {
        'f_t': {'format':
                   '\n %(asctime)s | %(levelname)s | %(name)s \t %(message)s'}

    },
    handlers = {
        'stream_handler':{
            'class': 'logging.StreamHandler',
            'formatter': 'f_t',
            'level': logging.INFO},
        'file_handler':{
            'class': 'logging.FileHandler',
            'formatter': 'f_t',
            'level': logging.INFO,
            'filename': None,
        }
    },
    root = {
        'handlers': ['stream_handler', 'file_handler'],
        'level': logging.DEBUG,
    },
)


def tfb_add_scalar_all_cls(tbwriter=None, type='train', epoch=None, CLASS_NAMES=None, data=None, datatype='accuracy'):

    for idx, cls in enumerate(CLASS_NAMES):
        tbwriter.add_scalar('{}_{}/{}'.format(type, datatype, cls), data[idx], epoch)

def create_logger(cfg, postfix=''):
    """Set up the logger for saving log file on the disk

    Args:
        cfg: configuration dict
        postfix: postfix of the log file name
    
    Return:
        logger: a logger for record essential information
    """
    # set up logger
    log_file = '{}_{}_{}.log'.format(cfg.network.name, cfg.dataset.name, postfix+time.strftime('%Y-%m-%d-%H-%M'))
    log_file_path = os.path.join(cfg.savedir, log_file)

    logging_config['handlers']['file_handler']['filename'] = log_file_path

    open(log_file_path,'w').close() #Clear the content of logfile
    # get logger from dictConfig
    dictConfig(logging_config)

    logger = logging.getLogger()

    return logger

def random_init(seed=0):
    """Set the seed for the random for torch and random package
    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def check_para_correctness(cfg):
    if not os.path.exists(cfg.savedir):
        os.makedirs(cfg.savedir)
    assert os.path.exists(cfg.savedir), '{} does not exist'.format(cfg.savedir)

    # if not os.path.exists(cfg.checkpointdir):
    #     os.makedirs(osp.join('/p300/PycharmProjects/ChestXrayCls', cfg.checkpointdir))

def get_inverse_images(image):
    mean_bgr = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_bgr = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = image.detach().cpu().numpy()
    image = image.transpose(0,2,3,1)
    image *= std_bgr
    image += mean_bgr
    image *= 255.0
    image = image[:, :, :, [2,1,0]]
    return image