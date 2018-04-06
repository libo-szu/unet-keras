
"""



new_unet:先三分支后三分支融合形成第四分支，后融合形成conv10卷积层






"""



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
#from keras import backend as K
from data import load_train_data, load_test_data

#from skimage.transform import rotate, resize
#from skimage import data
import matplotlib.pyplot as plt
import tensorflow as tf
from unet import get_unet
from model1 import craete_model1
from model2 import inception_model,create_model2
from model3 import inception_model1,create_model3
from new_model import new_model
from keras import backend as K
K.set_image_dim_ordering('th')  
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from train_test import dice_coef,dice_coef_loss

def new_unet(img_rows, img_cols):
	#第一分支

    inputs = Input((1, img_rows, img_cols))
    conv1_0 = Conv2D(32, 3,  activation='relu', padding='same')(inputs)
    conv1_0 = Conv2D(32, 3,  activation='relu', padding='same')(conv1_0)
    pool1_0 = MaxPooling2D(pool_size=(2, 2))(conv1_0)

    conv2_0 = Conv2D(48, 3, activation='relu', padding='same')(pool1_0)
    conv2_0 = Conv2D(48, 3, activation='relu', padding='same')(conv2_0)
    pool2_0 = MaxPooling2D(pool_size=(2, 2))(conv2_0)

    conv3_0 = Conv2D(72,3, activation='relu', padding='same')(pool2_0)
    conv3_0 = Conv2D(72, 3, activation='relu', padding='same')(conv3_0)
    pool3_0 = MaxPooling2D(pool_size=(2, 2))(conv3_0)

    conv4_0=Conv2D(128, 3, activation='relu', padding='same')(pool3_0)
    conv4_0 = Conv2D(128, 3, activation='relu', padding='same')(conv4_0)
    pool4_0 = MaxPooling2D(pool_size=(2, 2))(conv4_0)

    conv5_0 = Conv2D(256, 3, activation='relu', padding='same')(pool4_0)
    conv5_0 = Conv2D(256, 3, activation='relu', padding='same')(conv5_0)
    up6_0 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv5_0), conv4_0], mode='concat', concat_axis=1)
    conv6_0	= Conv2D(128, 3, activation='relu', padding='same')(up6_0)
    conv6_0 = Conv2D(128, 3, activation='relu', padding='same')(conv6_0)

    up7_0 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv6_0), conv3_0], mode='concat', concat_axis=1)
    conv7_0 = Conv2D(72, 3, activation='relu', padding='same')(up7_0)
    conv7_0 = Conv2D(72, 3, activation='relu', padding='same')(conv7_0)

    up8_0= merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv7_0), conv2_0], mode='concat', concat_axis=1)
    conv8_0 = Conv2D(48, 3, activation='relu', padding='same')(up8_0)
    conv8_0 = Conv2D(48, 3, activation='relu', padding='same')(conv8_0)

    up9_0 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv8_0), conv1_0], mode='concat', concat_axis=1)
    conv9_0 = Conv2D(32, 3, activation='relu', padding='same')(up9_0)
    conv9_0 = Conv2D(32, 3, activation='relu', padding='same')(conv9_0) 
	#第二分支
	
	
    conv1_1 = Conv2D(32, 5, activation='relu', padding='same')(inputs)
    conv1_1 = Conv2D(32,5, activation='relu', padding='same')(conv1_1)
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv2_1 = Conv2D(48, 5, activation='relu', padding='same')(pool1_1)
    conv2_1 = Conv2D(48, 5, activation='relu', padding='same')(conv2_1)
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    conv3_1	= Conv2D(72,5, activation='relu', padding='same')(pool2_1)
    conv3_1 = Conv2D(72, 5, activation='relu', padding='same')(conv3_1)
    pool3_1 = MaxPooling2D(pool_size=(2, 2))(conv3_1)

    conv4_1 = Conv2D(128, 5, activation='relu', padding='same')(pool3_1)
    conv4_1 = Conv2D(128, 5, activation='relu', padding='same')(conv4_1)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_1)

    conv5_1=Conv2D(256, 5, activation='relu', padding='same')(pool4_1)
    conv5_1 = Conv2D(256, 5, activation='relu', padding='same')(conv5_1)
    up6_1 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv5_1), conv4_1], mode='concat', concat_axis=1)
    conv6_1 = Conv2D(128, 5, activation='relu', padding='same')(up6_1)
    conv6_1 = Conv2D(128, 5, activation='relu', padding='same')(conv6_1)

    up7_1 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv3_1], mode='concat', concat_axis=1)
    conv7_1 = Conv2D(72, 5, activation='relu', padding='same')(up7_1)
    conv7_1 = Conv2D(72, 5, activation='relu', padding='same')(conv7_1)

    up8_1 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv7_1), conv2_1], mode='concat', concat_axis=1)
    conv8_1 = Conv2D(48, 5, activation='relu', padding='same')(up8_1)
    conv8_1 = Conv2D(48, 5, activation='relu', padding='same')(conv8_1)

    up9_1 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv8_1), conv1_1], mode='concat', concat_axis=1)
    conv9_1 = Conv2D(32, 5, activation='relu', padding='same')(up9_1)
    conv9_1 = Conv2D(32, 5, activation='relu', padding='same')(conv9_1) 
	
	
    #第三分支
	
	
    conv1_2 = Conv2D(32, 7, activation='relu', padding='same')(inputs)
    conv1_2 = Conv2D(32,7, activation='relu', padding='same')(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv2_2 = Conv2D(48, 7, activation='relu', padding='same')(pool1_2)
    conv2_2 = Conv2D(48, 7, activation='relu', padding='same')(conv2_2)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    conv3_2	= Conv2D(72,7, activation='relu', padding='same')(pool2_2)
    conv3_2 = Conv2D(72, 7, activation='relu', padding='same')(conv3_2)
    pool3_2 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

    conv4_2 = Conv2D(128, 7, activation='relu', padding='same')(pool3_2)
    conv4_2 = Conv2D(128, 7, activation='relu', padding='same')(conv4_2)
    pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_2)

    conv5_2 = Conv2D(256, 7, activation='relu', padding='same')(pool4_2)
    conv5_2 = Conv2D(256, 7, activation='relu', padding='same')(conv5_2)
    up6_2 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv5_2),conv4_2], mode='concat', concat_axis=1)
    conv6_2 = Conv2D(128, 7, activation='relu', padding='same')(up6_2)
    conv6_2 = Conv2D(128, 7, activation='relu', padding='same')(conv6_2)

    up7_2 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3_2], mode='concat', concat_axis=1)
    conv7_2 = Conv2D(72, 7, activation='relu', padding='same')(up7_2)
    conv7_2 = Conv2D(72, 7, activation='relu', padding='same')(conv7_2)

    up8_2 = merge([Conv2DT(256, (2, 2), strides=(2, 2), padding='same')(conv7_2), conv2_2], mode='concat', concat_axis=1)
    conv8_2 = Conv2D(48, 7, activation='relu', padding='same')(up8_2)
    conv8_2 = Conv2D(48, 7, activation='relu', padding='same')(conv8_2)

    up9_2 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv8_2), conv1_2], mode='concat', concat_axis=1)
    conv9_2 = Conv2D(32, 7, activation='relu', padding='same')(up9_2)
    conv9_2 = Conv2D(32, 7, activation='relu', padding='same')(conv9_2) 
	
	
    
	
	#第四分支
    conv5=merge([conv5_2,conv5_0,conv5_1],mode='concat',concat_axis=1)
    up6_3 = merge([Conv2D(128, (2, 2), strides=(2, 2), padding='same')(conv5),conv4_2,conv4_1,conv4_0], mode='concat', concat_axis=1)
    conv6_3 = Conv2D(128, 3, activation='relu', padding='same')(up6_3)
    conv6_3 = Conv2D(128, 3, activation='relu', padding='same')(conv6_3)

    up7_3 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv6_3), conv3_0,conv3_1,conv3_2], mode='concat', concat_axis=1)
    conv7_3 = Conv2D(72, 3, activation='relu', padding='same')(up7_3)
    conv7_3 = Conv2D(72, 3, activation='relu', padding='same')(conv7_3)

    up8_3 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv7_3), conv2_0,conv2_1,conv2_2], mode='concat', concat_axis=1)
    conv8_3 = Conv2D(48, 3, activation='relu', padding='same')(up8_3)
    conv8_3 = Conv2D(48, 3, activation='relu', padding='same')(conv8_3)

    up9_3 = merge([Conv2D(256, (2, 2), strides=(2, 2), padding='same')(conv8_3), conv1_0,conv1_1,conv1_2], mode='concat', concat_axis=1)
    conv9_3 = Conv2D(32, 3, activation='relu', padding='same')(up9_3)
    conv9_3 = Conv2D(32, 3, activation='relu', padding='same')(conv9_3) 
	
	
	
	#四个分支融合
    conv10=merge([conv9_3,conv9_1,conv9_0,conv9_2],mode='concat',concat_axis=1)
    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv10)
    model = Model(input=inputs, output=conv10)
   
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model  
