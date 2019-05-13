#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019-04-28 10:12
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import numpy as np
import csv
import os.path as osp
import os
import json
import pandas as pd
import sys

CLASS_NAMES = [ 'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass',
                'Nodule', 'Atelectasis', 'Pneumothorax',  'Pleural_Thickening', 'Pneumonia',
                'Fibrosis', 'Edema', 'Consolidation']

def dataset_split(csv_file=None):

    with open(osp.join('./data/nih_labels.txt'), 'r') as file:
        file_list = file.readlines()

    image_name_list = []
    label_list = []
    data_type_list = []

    training_img_name = []
    val_img_name = []
    test_img_name = []

    training_label = []
    val_label = []
    test_label = []

    for line in file_list:
        items = line.split(',')
        data_type = items[-1].strip()
        data_type_list.append(items[-1].strip())
        # image_name_list.append(items[0])
        label = []
        for diseases_id in range(14):
            label.append(float(items[6+diseases_id]))
       # label_list.append(label)

        if data_type == 'train':
            training_img_name.append(items[0])
            training_label.append(label)
        elif data_type=='val':
            val_img_name.append(items[0])
            val_label.append(label)
        elif data_type=='test':
            test_img_name.append(items[0])
            test_label.append(label)


    # print(len(image_name_list))
    # print(np.array(label_list).shape)
    # print(len(data_type_list))
    # train_index = np.where((np.array(data_type_list)=='train'))[0].tolist()
    # val_index = np.where((np.array(data_type_list)=='val'))[0].tolist()
    # test_index = np.where((np.array(data_type_list)=='test'))[0].tolist()
    #
    # image_name_list = np.array(image_name_list)
    # label_list = np.array(label_list)
    training_json = {'image_name': training_img_name}
    val_json = {'image_name': val_img_name}
    test_json = {'image_name': test_img_name}
    #
    # print(label_list[train_index].shape)
    # print(label_list[val_index].shape)
    # print(label_list[test_index].shape)
    training_json['label'] = training_label
    val_json['label'] = val_label
    test_json['label'] = test_label
    print(np.array(training_label).shape)
    print(np.array(val_label).shape)
    print(np.array(test_label).shape)

    with open('./data/processed/labels/training_label.json', 'w') as dump_f:
        json.dump(training_json, dump_f)

    with open('./data/processed/labels/val_label.json', 'w') as dump_f:
        json.dump(val_json, dump_f)

    with open('./data/processed/labels/test_label.json', 'w') as dump_f:
        json.dump(test_json, dump_f)

    # with open('./data/processed/labels/test_label.json', 'r') as load_f:
    #     test_json = json.load(load_f)
    #
    # print(np.array(test_json['label']).shape)

def read_data_entry():

    with open('./data/Data_Entry_2017.txt', 'r') as f:
        dataset = f.readlines()

    label_dict = {}
    for line in dataset:
        line = line.strip().split(',')
        img_name = line[0]
        label_diseases = line[1].split('|')
        label_one_hot = [0]*14
        if 'No Finding' in label_diseases:
            label_dict[img_name] = label_one_hot
        else:
            for label in label_diseases:
                label_idx = CLASS_NAMES.index(label)
                label_one_hot[label_idx] = 1

            label_dict[img_name] = label_one_hot
    with open('./data/processed/labels/Data_Entry_2017.json', 'w') as dump_f:
        json.dump(label_dict, dump_f)


def sample_val_dataset():

    with open('./data/train_val_list.txt', 'r') as read_f:
        train_all = read_f.readlines()

    train = open('./data/processed/labels/training_list.txt', 'w')
    val = open('./data/processed/labels/val_list.txt', 'w')
    trainset_id = []
    valset_id = []
    data_exist_set = []
    for img_name in train_all:
        rand_seed = np.random.random()
        img_name = img_name.strip()
        img_id = img_name.split('_')[0]
        if img_id not in data_exist_set:
            if rand_seed > 0.13:
                trainset_id.append(img_id)
            else:
                valset_id.append(img_id)
            data_exist_set.append(img_id)

    print(len(trainset_id))
    print(len(valset_id))
    train_count = 0
    val_count = 0
    for img_name in train_all:
        img_name = img_name.strip()
        img_id = img_name.split('_')[0]
        if img_id in trainset_id:
            train.write('{}\n'.format(img_name))
            train_count += 1
        elif img_id in valset_id:
            val_count += 1
            val.write('{}\n'.format(img_name))

    print(train_count)
    print(val_count)
    train.close()
    val.close()


def sample_mini_train_val():

    with open('./data/processed/labels/training_list.txt', 'r') as read_f:
        training_list = read_f.readlines()

    with open('./data/processed/labels/val_list.txt', 'r') as read_f:
        val_list = read_f.readlines()

    training_mini = open('./data/processed/labels/training_mini_list.txt', 'w')
    val_mini = open('./data/processed/labels/val_mini_list.txt', 'w')

    for line in training_list:
        line = line.strip()
        if np.random.random()<0.1:
            training_mini.write(line+'\n')

    for line in val_list:
        line = line.strip()
        if np.random.random() < 0.25:
            val_mini.write(line + '\n')
    training_mini.close()
    val_mini.close()




if __name__ == '__main__':
    # dataset_split()
    # read_data_entry()
    np.random.seed(10)
    # sample_val_dataset()
    sample_mini_train_val()

