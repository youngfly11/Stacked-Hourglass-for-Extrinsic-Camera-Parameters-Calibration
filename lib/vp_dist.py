#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019-05-16 00:41
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import numpy as np
import os.path as osp
import os
import cv2
import matplotlib.pyplot as plt
import imageio


def draw_vp():

    with open('./data/processed/labels/training.txt', 'r') as read_f:
        file = read_f.readlines()

    fake_img = np.ones((480, 640, 3))*255
    fake_img = fake_img.astype(np.float32)
    plt.imshow(fake_img)

    for idx, line in enumerate(file):

        line = line.split(',')
        x_label = int(line[3])
        y_label = int(line[4])
        print(x_label, y_label)
        fake_img = cv2.circle(fake_img, center=(x_label, y_label), radius=4, color=(255, 0,0), thickness=1)

    imageio.imsave('dist.png', fake_img)
    plt.imshow(fake_img)
    plt.show()

if __name__ == '__main__':
    draw_vp()




