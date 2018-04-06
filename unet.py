#原始unet模型 

from __future__ import print_function
from keras import backend as K
import numpy as np
from keras.models import Model,load_model
from keras.layers import Input, merge, Dropout
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,UpSampling2D
from keras.layers.pooling  import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import matplotlib.pyplot as plt
import tensorflow as tf
from train_test import dice_coef_loss,dice_coef
from keras import backend as K
K.set_image_dim_ordering('th')  


"""这是一个原始unet的实现模型"""
def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Conv2D(32, 3, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, 3, activation='relu', padding='same')(conv5)
    # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(convdeep)
    
    # upmid = merge([Convolution2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(convdeep)), conv5], mode='concat', concat_axis=1)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(upmid)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(convmid)

    up6 = merge([Conv2D(256, 2, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5)), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, 3, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, 3, 3, activation='relu', padding='same')(conv6)

    up7 = merge([Conv2D(128, 2, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2D(128, 3, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, 3, 3, activation='relu', padding='same')(conv7)

    up8 = merge([Conv2D(64, 2, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7)), conv2], mode='concat', concat_axis=1)
    conv8 = Conv2D(64, 3, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, 3, 3, activation='relu', padding='same')(conv8)

    up9 = merge([Conv2D(32, 2, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8)), conv1], mode='concat', concat_axis=1)
    conv9 = Conv2D(32, 3, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, 3, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model
