from __future__ import print_function
from keras import backend as K
import cv2
import numpy as np
from keras.models import Model,load_model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from data import load_train_data, load_test_data

from skimage.transform import rotate, resize
from skimage import data
import matplotlib.pyplot as plt
import tensorflow as tf


from keras import backend as K

"""这是一个打印观察训练数据的文件，并非测试文件"""
def test():
    imgs_train=np.load("imgs_train.npy")
    imgs_mask_train=np.load("data.npy")
    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    for img in imgs_mask_train:
        plt.imshow(img[0,:,:])
        plt.show()
if __name__ == '__main__':
    test()
