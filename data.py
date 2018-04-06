from __future__ import print_function

import os
import numpy as np

import cv2
import pandas as pd
import sys
import os
import os.path
import string
import scipy.io
import pdb
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

"""这是一个将图像文件转换成numpy数据的py代码"""

data_path = 'raw1/'
save_path = 'C:/Users/server/Desktop/u-net-master/u-net-master/'
image_rows = 224
image_cols = 224
#创建训练数据
def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array(img)
        img_mask = np.array(img_mask)

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')

   #下载训练数据的函数
def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train
#对数据进行resize
def preprocess(imgs, img_rows,img_cols):
    imgs_p = np.ndarray((imgs.shape[0],imgs.shape[1],img_rows,img_cols),dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0 ] = cv2.resize(imgs[i,0],(img_cols,img_rows),interpolation=cv2.INTER_CUBIC)
    return imgs_p
#创建最终的训练数据：data.npy,mask.npy
def detseg():
    # out_rows=240
    # out_cols=320
    out_rows=224
    out_cols=224
    imgs_train = np.load('imgs_train.npy')
    imgs_train=preprocess(imgs_train, out_rows,out_cols).astype(np.float32)
    #mean_image=imgs_train.mean(0)[np.newaxis,]
    #imgs_train -=mean_image
    print(np.histogram(imgs_train))
    #std_image=imgs_train.std(0)[np.newaxis,]
    #imgs_train /=std_image
    print(np.histogram(imgs_train))

    imgs_mask_train = np.load('imgs_mask_train.npy')
    imgs_mask_train=preprocess(imgs_mask_train, out_rows,out_cols)
    imgs_mask_train[imgs_mask_train<=50]=False
    imgs_mask_train[imgs_mask_train>50]=True
    print(np.histogram(imgs_mask_train))

    # if os.path.exists(save_path+'data.npy')==False:
   # np.save(save_path+'mean.npy',mean_image)
   # np.save(save_path+'std.npy',std_image)
    np.save(save_path+'data.npy',imgs_train.astype(np.float32))
    print('save data')
    np.save(save_path+'mask.npy',imgs_mask_train.astype(np.bool))
    print('save mask')
#创建test数据
def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_mask_test.npy', imgs_mask)
    print('Saving to .npy files done.')
#下载test数据
def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_mask_test = np.load('imgs_mask_test.npy')
    return imgs_test, imgs_mask_test

if __name__ == '__main__':
    #create_train_data()
    #create_test_data()
    detseg()
