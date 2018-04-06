from __future__ import print_function
from keras import backend as K
import cv2
import numpy as np
from keras.models import Model,load_model
from keras.layers import Input, merge, Dropout
from keras.layers.convolutional import Conv2D,UpSampling2D
from keras.layers.pooling  import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from data import load_train_data, load_test_data

#from skimage.transform import rotate, resize
#from skimage import data
import matplotlib.pyplot as plt
import tensorflow as tf


from keras import backend as K
K.set_image_dim_ordering('th')
img_rows = 224
img_cols = 224

smooth = 1.



"""
unet1.hdf5:三分支网络权重文件

unet.hdf5：原始unet网络权重文件
unet_change:先分后和模型







"""

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def inception_model(inputs,kernel_size):
    conv1 = Conv2D(32, kernel_size, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32,kernel_size, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kernel_size, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kernel_size, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128,kernel_size, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kernel_size, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kernel_size, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kernel_size, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, kernel_size, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, kernel_size, activation='relu', padding='same')(conv5)
    up6 = merge([Conv2D(256, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5)), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, kernel_size, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, kernel_size, activation='relu', padding='same')(conv6)

    up7 = merge([Conv2D(128, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2D(128, kernel_size, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, kernel_size, activation='relu', padding='same')(conv7)

    up8 = merge([Conv2D(64, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7)), conv2], mode='concat', concat_axis=1)
    conv8 = Conv2D(64, kernel_size, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, kernel_size, activation='relu', padding='same')(conv8)

    up9 = merge([Conv2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8)), conv1], mode='concat', concat_axis=1)
    conv9 = Conv2D(32, kernel_size, activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, kernel_size, activation='relu', padding='same')(conv9) 
    return conv9	

	
	
	

	
def inception_model1(inputs,kernel_size):
    conv1 = Conv2D(32, (kernel_size,3) ,activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32,(kernel_size,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (kernel_size,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (kernel_size,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128,(kernel_size,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (kernel_size,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (kernel_size,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (kernel_size,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (kernel_size,3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (kernel_size,3), activation='relu', padding='same')(conv5)
    up6 = merge([Conv2D(256, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5)), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, (kernel_size,3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (kernel_size,3), activation='relu', padding='same')(conv6)

    up7 = merge([Conv2D(128, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2D(128, (kernel_size,3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (kernel_size,3), activation='relu', padding='same')(conv7)

    up8 = merge([Conv2D(64, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7)), conv2], mode='concat', concat_axis=1)
    conv8 = Conv2D(64, (kernel_size,3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (kernel_size,3), activation='relu', padding='same')(conv8)

    up9 = merge([Conv2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8)), conv1], mode='concat', concat_axis=1)
    conv9 = Conv2D(32, (kernel_size,3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (kernel_size,3), activation='relu', padding='same')(conv9) 
    return conv9
"""
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
"""
#三分支后融合
def craete_model1():
    inputs = Input((1, img_rows, img_cols))
    conv1_0 = Conv2D(32, 3,  activation='relu', padding='same')(inputs)
    conv1_0 = Conv2D(32, 3,  activation='relu', padding='same')(conv1_0)
    pool1_0 = MaxPooling2D(pool_size=(2, 2))(conv1_0)
    
    conv1_1 = Conv2D(32, 5, activation='relu', padding='same')(inputs)
    conv1_1 = Conv2D(32, 5, activation='relu', padding='same')(conv1_1)
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)  


	
    conv1_2 = Conv2D(32, 7, activation='relu', padding='same')(inputs)
    conv1_2 = Conv2D(32,7, activation='relu', padding='same')(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2) 

	
    conv2_0 = Conv2D(64, 3,  activation='relu', padding='same')(pool1_0)
    conv2_0 = Conv2D(64, 3, activation='relu', padding='same')(conv2_0)
    pool2_0 = MaxPooling2D(pool_size=(2, 2))(conv2_0)
	
	
	
    conv2_1 = Conv2D(64, 5, activation='relu', padding='same')(pool1_1)
    conv2_1 = Conv2D(64, 5, activation='relu', padding='same')(conv2_1)
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
	


    conv2_2 = Conv2D(64, 7, activation='relu', padding='same')(pool1_2)
    conv2_2 = Conv2D(64, 7, activation='relu', padding='same')(conv2_2)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)	
	
	
	
    conv3_0 = Conv2D(128, 3, activation='relu', padding='same')(pool2_0)
    conv3_0 = Conv2D(128, 3, activation='relu', padding='same')(conv3_0)
    pool3_0 = MaxPooling2D(pool_size=(2, 2))(conv3_0)
	
	
    conv3_1 = Conv2D(128, 5, activation='relu', padding='same')(pool2_1)
    conv3_1 = Conv2D(128, 5, activation='relu', padding='same')(conv3_1)
    pool3_1 = MaxPooling2D(pool_size=(2, 2))(conv3_1)
    

    conv3_2 = Conv2D(128, 7, activation='relu', padding='same')(pool2_2)
    conv3_2 = Conv2D(128, 7, activation='relu', padding='same')(conv3_2)
    pool3_2 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
	
	
	
	
	
    conv4_0 = Conv2D(256, 3, activation='relu', padding='same')(pool3_0)
    conv4_0 = Conv2D(256, 3, activation='relu', padding='same')(conv4_0)
    pool4_0 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
	
	
    conv4_1 = Conv2D(256, 5, activation='relu', padding='same')(pool3_1)
    conv4_1 = Conv2D(256, 5, activation='relu', padding='same')(conv4_1)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_1)

	
    conv4_2 = Conv2D(256, 7, activation='relu', padding='same')(pool3_2)
    conv4_2 = Conv2D(256, 7, activation='relu', padding='same')(conv4_2)
    pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_2)
    

    
	
	


    conv5_0 = Conv2D(512, 3, activation='relu', padding='same')(pool4_0)
    conv5_0 = Conv2D(512, 3, activation='relu', padding='same')(conv5_0)
    #pool5_0 = MaxPooling2D(pool_size=(2, 2))(conv5)
	
	
    conv5_1 = Conv2D(512, 5, activation='relu', padding='same')(pool4_1)
    conv5_1 = Conv2D(512, 5, activation='relu', padding='same')(conv5_1)
    #pool5_1 = MaxPooling2D(pool_size=(2, 2))(conv5)
	

	
    
    conv5_2 = Conv2D(512, 7, activation='relu', padding='same')(pool4_2)
    conv5_2 = Conv2D(512, 7, activation='relu', padding='same')(conv5_2)
    #pool5_2 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv5=merge([conv5_0,conv5_1,conv5_2],mode='concat', concat_axis=1)
	
    

	
    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(convdeep)
    
    # upmid = merge([Convolution2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(convdeep)), conv5], mode='concat', concat_axis=1)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(upmid)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(convmid)

    up6 = merge([Conv2D(256, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5)), conv4_0,conv4_1,conv4_2], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = merge([Conv2D(128, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3_0,conv3_1,conv3_2], mode='concat', concat_axis=1)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = merge([Conv2D(64, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7)), conv2_0,conv2_1,conv2_2], mode='concat', concat_axis=1)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up9 = merge([Conv2D(32, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8)), conv1_0,conv1_1,conv1_2], mode='concat', concat_axis=1)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, 3,  activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model
	
#正方形卷积核    三独立分支	
def create_model2():
    inputs = Input((1, img_rows, img_cols))
    inception1=inception_model(inputs,3)
    inception2=inception_model(inputs,5)
    inception3=inception_model(inputs,7)
    conv1=merge([inception1,inception2,inception3],mode='concat', concat_axis=1)
    conv2 = Conv2D(1, 1, 1, activation='sigmoid')(conv1)
    model = Model(input=inputs, output=conv2)
   
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model    
    	


#长方形卷积核   三独立分支	
def create_model3():
    inputs = Input((1, img_rows, img_cols))
    inception1=inception_model1(inputs,3)
    inception2=inception_model1(inputs,5)
    inception3=inception_model1(inputs,7)
    conv1=merge([inception1,inception2,inception3],mode='concat', concat_axis=1)
    conv2 = Conv2D(1, 1, 1, activation='sigmoid')(conv1)
    model = Model(input=inputs, output=conv2)
   
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model  
def new_model():
    inputs = Input((1, img_rows, img_cols))
    conv1_0 = Conv2D(32, 3,  activation='relu', padding='same')(inputs)
    conv1_0 = Conv2D(32, 3,  activation='relu', padding='same')(conv1_0)
    pool1_0 = MaxPooling2D(pool_size=(2, 2))(conv1_0)
    
    conv1_1 = Conv2D(32, 5, activation='relu', padding='same')(inputs)
    conv1_1 = Conv2D(32, 5, activation='relu', padding='same')(conv1_1)
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)  


	
    conv1_2 = Conv2D(32, 7, activation='relu', padding='same')(inputs)
    conv1_2 = Conv2D(32,7, activation='relu', padding='same')(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2) 

	
    conv2_0 = Conv2D(64, 3,  activation='relu', padding='same')(pool1_0)
    conv2_0 = Conv2D(64, 3, activation='relu', padding='same')(conv2_0)
    pool2_0 = MaxPooling2D(pool_size=(2, 2))(conv2_0)
	
	
	
    conv2_1 = Conv2D(64, 5, activation='relu', padding='same')(pool1_1)
    conv2_1 = Conv2D(64, 5, activation='relu', padding='same')(conv2_1)
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
	


    conv2_2 = Conv2D(64, 7, activation='relu', padding='same')(pool1_2)
    conv2_2 = Conv2D(64, 7, activation='relu', padding='same')(conv2_2)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)	
	
	
	
    conv3_0 = Conv2D(128, 3, activation='relu', padding='same')(pool2_0)
    conv3_0 = Conv2D(128, 3, activation='relu', padding='same')(conv3_0)
    pool3_0 = MaxPooling2D(pool_size=(2, 2))(conv3_0)
	
	
    conv3_1 = Conv2D(128, 5, activation='relu', padding='same')(pool2_1)
    conv3_1 = Conv2D(128, 5, activation='relu', padding='same')(conv3_1)
    pool3_1 = MaxPooling2D(pool_size=(2, 2))(conv3_1)
    

    conv3_2 = Conv2D(128, 7, activation='relu', padding='same')(pool2_2)
    conv3_2 = Conv2D(128, 7, activation='relu', padding='same')(conv3_2)
    pool3_2 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
	
	
	
	
	
    conv4_0 = Conv2D(256, 3, activation='relu', padding='same')(pool3_0)
    conv4_0 = Conv2D(256, 3, activation='relu', padding='same')(conv4_0)
    pool4_0 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
	
	
    conv4_1 = Conv2D(256, 5, activation='relu', padding='same')(pool3_1)
    conv4_1 = Conv2D(256, 5, activation='relu', padding='same')(conv4_1)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_1)

	
    conv4_2 = Conv2D(256, 7, activation='relu', padding='same')(pool3_2)
    conv4_2 = Conv2D(256, 7, activation='relu', padding='same')(conv4_2)
    pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_2)
    

    
	
	


    conv5_0 = Conv2D(512, 3, activation='relu', padding='same')(pool4_0)
    conv5_0 = Conv2D(512, 3, activation='relu', padding='same')(conv5_0)
    #pool5_0 = MaxPooling2D(pool_size=(2, 2))(conv5)
	
	
    conv5_1 = Conv2D(512, 5, activation='relu', padding='same')(pool4_1)
    conv5_1 = Conv2D(512, 5, activation='relu', padding='same')(conv5_1)

    
    conv5_2 = Conv2D(512, 7, activation='relu', padding='same')(pool4_2)
    conv5_2 = Conv2D(512, 7, activation='relu', padding='same')(conv5_2)
    #pool5_2 = MaxPooling2D(pool_size=(2, 2))(conv5)

    up6_0 = merge([Conv2D(256, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5_0)),conv4_1,conv4_2], mode='concat', concat_axis=1)
    conv6_0 = Conv2D(256, 3, activation='relu', padding='same')(up6_0)
    conv6_0 = Conv2D(256, 3, activation='relu', padding='same')(conv6_0)
	
	
    up6_1 = merge([Conv2D(256, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5_1)),conv4_0,conv4_2], mode='concat', concat_axis=1)
    conv6_1 = Conv2D(256, 3, activation='relu', padding='same')(up6_1)
    conv6_1 = Conv2D(256, 3, activation='relu', padding='same')(conv6_1)

	
    up6_2 = merge([Conv2D(256, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5_2)),conv4_0,conv4_1], mode='concat', concat_axis=1)
    conv6_2 = Conv2D(256, 3, activation='relu', padding='same')(up6_2)
    conv6_2 = Conv2D(256, 3, activation='relu', padding='same')(conv6_2)

    up7_0 = merge([Conv2D(128, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6_0)),conv3_1,conv3_2], mode='concat', concat_axis=1)
    conv7_0 = Conv2D(128, 3, activation='relu', padding='same')(up7_0)
    conv7_0 = Conv2D(128, 3, activation='relu', padding='same')(conv7_0)
	
	
    up7_1 = merge([Conv2D(128, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6_1)),conv3_0,conv3_2], mode='concat', concat_axis=1)
    conv7_1 = Conv2D(128, 3, activation='relu', padding='same')(up7_1)
    conv7_1 = Conv2D(128, 3, activation='relu', padding='same')(conv7_1)
   
   
    up7_2 = merge([Conv2D(128, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6_2)),conv3_1,conv3_0], mode='concat', concat_axis=1)
    conv7_2 = Conv2D(128, 3, activation='relu', padding='same')(up7_2)
    conv7_2 = Conv2D(128, 3, activation='relu', padding='same')(conv7_2)

    up8_0 = merge([Conv2D(64, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7_0)), conv2_1,conv2_2], mode='concat', concat_axis=1)
    conv8_0 = Conv2D(64, 3, activation='relu', padding='same')(up8_0)
    conv8_0 = Conv2D(64, 3, activation='relu', padding='same')(conv8_0)

	
	
    up8_1 = merge([Conv2D(64, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7_1)), conv2_0,conv2_2], mode='concat', concat_axis=1)
    conv8_1 = Conv2D(64, 3, activation='relu', padding='same')(up8_1)
    conv8_1 = Conv2D(64, 3, activation='relu', padding='same')(conv8_1)
	
	
    up8_2 = merge([Conv2D(64, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7_2)), conv2_1,conv2_0], mode='concat', concat_axis=1)
    conv8_2 = Conv2D(64, 3, activation='relu', padding='same')(up8_2)
    conv8_2 = Conv2D(64, 3, activation='relu', padding='same')(conv8_2)



    up9_0 = merge([Conv2D(32, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8_0)),conv1_1,conv1_2], mode='concat', concat_axis=1)
    conv9_0 = Conv2D(32, 3, activation='relu', padding='same')(up9_0)
    conv9_0 = Conv2D(32, 3,  activation='relu', padding='same')(conv9_0)
	
	
	
	

    up9_1 = merge([Conv2D(32, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8_1)),conv1_0,conv1_2], mode='concat', concat_axis=1)
    conv9_1 = Conv2D(32, 3, activation='relu', padding='same')(up9_1)
    conv9_1 = Conv2D(32, 3,  activation='relu', padding='same')(conv9_1)
 
 
 

    up9_2 = merge([Conv2D(32, 2,activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8_2)),conv1_1,conv1_0], mode='concat', concat_axis=1)
    conv9_2 = Conv2D(32, 3, activation='relu', padding='same')(up9_2)
    conv9_2 = Conv2D(32, 3,  activation='relu', padding='same')(conv9_2)
	
	
	
    conv9=merge([conv9_0,conv9_1,conv9_2],mode='concat',concat_axis=1)
	

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model















 	
def train():
    print('-'*30)
    print('Loading  train data...')
    print('-'*30)
   
    imgs_train=np.load("D:/DSB2018_03_18/npyFile/trainIMG.npy").transpose((0,3,1,2))
    imgs_mask_train=np.load("D:/DSB2018_03_18/npyFile/trainMASK.npy").transpose((0,3,1,2))
    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    #imgs_test, imgs_mask_test = load_test_data()
    
    #imgs_test = imgs_test.astype('float32')
    #imgs_mask_test=imgs_mask_test.astype('float32')

    total=imgs_train.shape[0]
  

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = new_model()
    
    model_checkpoint = ModelCheckpoint('new_model1.hdf5', monitor='loss',verbose=1, save_best_only=True)

    
    model.fit(imgs_train, imgs_mask_train, batch_size=10, nb_epoch=120, verbose=1, shuffle=True,callbacks=[model_checkpoint])
    
    print('-'*30)
    print('训练完毕')
    print(*'-'*30)
	

""" 
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('unetfangxing.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test1 = model.predict(imgs_test, verbose=1)
	
  for i  in range(1165):
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.imshow(imgs_test[i,0,:,:],'gray')
        ax = fig.add_subplot(222)
        ax.imshow(imgs_mask_test[i,0,:,:],'gray') 
        ax = fig.add_subplot(223) 
        ax.imshow(imgs_mask_test1[i,0,:,:],'gray')  
        plt.show()
       
    np.save('net_model_imgs_mask_test.npy', imgs_mask_test1)
"""
def  test():
    #imgs_test1, imgs_mask_test = load_test_data()
    imgs_test=np.load('D:/DSB2018_03_18/npyFile/testIMG.npy').transpose((0,3,1,2))
    #print(imgs_test1.max(),imgs_test.max())
	#imgs_test=np.load('testIMG.npy').transpose((0,3,1,2))
    imgs_test = imgs_test.astype('float32')
	#三分支后融合
    model1 = new_model()
    model1.load_weights('new_model.hdf5')
	
	
    #model.load_weights('unet1.hdf5')

    #model = load_model('unet.hdf5', custom_objects={'dice_coef': dice_coef})
     #model = load_model('unet.hdf5')

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)


    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
	#环绕网络
    imgs_mask_test1 = model1.predict(imgs_test, verbose=1)
	#长方形卷积核
    model2=create_model3()
    model2.load_weights('unet1.hdf5')
    imgs_mask_test2=model2.predict(imgs_test*255, verbose=1)
	#正方形卷积核
    model3=create_model2()
    model3.load_weights('unetfangxing.hdf5')
    imgs_mask_test3=model3.predict(imgs_test*255, verbose=1)
    for i  in range(1165):
        fig = plt.figure()
		#打印原图
        ax = fig.add_subplot(221)
        plt.title('yuantu')
        ax.imshow(imgs_test[i,0,:,:],'gray')
        #ax = fig.add_subplot(222)
        #plt.title('mask')
        #ax.imshow(imgs_mask_test[i,0,:,:],'gray') 
		#打印环绕模型预测图
        ax = fig.add_subplot(222) 
        plt.title('huanrao')
        ax.imshow(imgs_mask_test1[i,0,:,:],'gray')
		#打印三分支模型长方形卷积核预测图
        ax = fig.add_subplot(223) 
        plt.title('sanfenzhi')
        ax.imshow(imgs_mask_test2[i,0,:,:],'gray')
		#打印三分支模型正方形卷积核预测图
        ax = fig.add_subplot(224) 
        plt.title('fangxingjuanji')
        ax.imshow(imgs_mask_test3[i,0,:,:],'gray')
        #ax.savefig('C:/Users/server/Desktop/u-net-master/u-net-master/先分后和模型test图像/'+str(i)+'.png')		
        plt.show()
    #np.save('imgs_mask_test.npy', imgs_mask_test)

if __name__ == '__main__':
    train()
    #test()
