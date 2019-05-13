#
# @Author: Songyang Zhang 
# @Date: 2018-11-16 11:57:39 
# @Last Modified by:   Songyang Zhang 
# @Last Modified time: 2018-11-16 11:57:39 
#

import torch
import numpy as np 
import cv2
import os
import time
from tqdm import tqdm
import pandas as pd

# @profile
def open_rgby(path, id):
    r"""
        Open RGBY Human Protein Image
    """

    colors = ['red', 'green', 'blue', 'yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = []
    for color in colors:
        temp = cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags)
        temp = temp.astype(np.float32)/255
        img.append(temp)

    # img = [.astype(np.float32)/255 for color in colors]

    return np.stack(img, axis=-1)


def processed_image():
    """
        Convert 4-channel image to .npy file
    """
    
    train_image_lists = list({f[:36] for f in os.listdir('./data/train')})
    test_image_lists = list({f[:36] for f in os.listdir('./data/test')})

    dataset_root_train = '/p300/Dataset/HumanProteinProcessed/train'
    if len(os.listdir(dataset_root_train)) == len(train_image_lists):
        print('Processed Already!')
    else:
        for step, item in enumerate(tqdm(train_image_lists)):
            img = open_rgby('./data/train', item) # (H, W, 4)
            np.save(os.path.join(dataset_root_train,item+'.npy'), img)
        print('Processed Done!')


    dataset_root_test = '/p300/Dataset/HumanProteinProcessed/test'
    if len(os.listdir(dataset_root_test)) == len(test_image_lists):
        print('Processed Already!')
    else:
        for item in test_image_lists:
            for color in ['red', 'green', 'blue', 'yellow']:
                assert os.path.exists(os.path.join('./data/test', item+'_'+color+'.png')), os.path.join('./data/test', item+'_'+color+'.png')
        import pdb; pdb.set_trace()
        for step, item in enumerate(tqdm(test_image_lists)):
            if os.path.exists(os.path.join(dataset_root_test,item+'.npy')):
                pass
            else:
                img = open_rgby('./data/test', item) # (H, W, 4)
                np.save(os.path.join(dataset_root_test,item+'.npy'), img)

def process_label():
    # import pdb; pdb.set_trace()
    train_image_lists = list({f[:36] for f in os.listdir('./data/train')})
    test_image_lists = list({f[:36] for f in os.listdir('./data/test')})

    len_labels = 28
    train_csv_file = 'data/train.csv'

    labels = pd.read_csv(train_csv_file).set_index('Id')
    labels['Target'] = [[int(i) for i in s.split()] for s in labels['Target']]

    labels_dict = {}
    for image_name in train_image_lists:
        label = labels.loc[image_name]['Target']
        label = np.eye(len_labels,dtype=np.float)[label].sum(axis=0)
        labels_dict[image_name] = label

    np.save('./data/train_labels.npy', labels_dict)
    
    # import pdb; pdb.set_trace()

process_label()