#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019-05-09 19:40
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import numpy as np
import os.path as osp
import os
import cv2
import imageio


def extract_url_and_label():

    file_data = open('./data/processed/labels/dataset.txt', 'w')
    file_data_img = open('./data/processed/labels/dataset.txt', 'w')

    base_path = './data/processed/VP_dataset'
    count = 0
    for road_type in os.listdir(base_path):

        file_path_road = osp.join(base_path, road_type)
        camera_id_road_list = os.listdir(file_path_road)

        for cam_id in camera_id_road_list:
            camera_img_list = os.listdir(osp.join(base_path, road_type, cam_id))

            for img_id in camera_img_list:
                with open(osp.join(base_path, road_type, cam_id, img_id)) as img_f:
                    file = img_f.readlines()
                for line in file:
                    line = line.strip()

                    # write_info_img_id = "{} {}"
                    write_info = '{}, {}\n'.format(img_id.split('.')[0], line)
                    file_data.write(write_info)

    print(count)

def dataset_split():

    training = open('./data/processed/labels/training.txt', 'w')
    val = open('./data/processed/labels/val.txt', 'w')
    test = open('./data/processed/labels/test.txt', 'w')

    with open('./data/processed/labels/dataset.txt', 'r') as load_f:
        f = load_f.readlines()

    for line in f:
        line = line.strip()
        rand_seed = np.random.random()
        if rand_seed >0.9:
            test.write(line+'\n')
        elif rand_seed >0.8:
            val.write(line+'\n')
        else:
            training.write(line+'\n')

def img_processing():

    base_path = '/Volumes/MyDisk/dataset/VP_Img'
    base_path1 = '/Volumes/MyDisk/dataset/VP_Img_resize'
    # base_path2 = '/Volumes/MyDisk/dataset/VP_Img_demo'
    base_path2 = '/Volumes/MyDisk/dataset/VP_Img_demo1'
    if not osp.exists(base_path1):
        os.makedirs(base_path1)
        os.makedirs(base_path2)


    with open('./data/processed/labels/dataset.txt', 'r') as load_f:
        f = load_f.readlines()

    for line in f:
        line = line.split(', ')
        img_name = line[0].split('_')
        gt_vp = (int(line[1]), int(line[2]))
        img = imageio.imread(osp.join(base_path, '{}/{}.jpg'.format(img_name[0], img_name[1])))
        img_resize = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
        path1 = osp.join(base_path1, img_name[0])
        path2 = osp.join(base_path2, img_name[0])
        if not osp.exists(path1):
            os.makedirs(path1)
        if not osp.exists(path2):
            os.makedirs(path2)
        imageio.imsave(osp.join(path1, '{}.jpg'.format(img_name[1])), img_resize)

        img_points = cv2.circle(img=img_resize, center=gt_vp, radius=5, color=(255,0,0))
        imageio.imsave(osp.join(path2, '{}.jpg'.format(img_name[1])), img_points)
        print('image {} done successfully'.format(line[0]))
        print(gt_vp)



if __name__ == '__main__':
    # extract_url_and_label()
    # dataset_split()
    img_processing()