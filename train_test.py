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
from unet import get_unet

from keras import backend as K
K.set_image_dim_ordering('th')  
from keras.preprocessing.image import ImageDataGenerator
from  model2 import create_model2
from  model3 import create_model3
from  model1 import craete_model1
from  unet import get_unet
from  new_unet  import new_unet
from new_model import new_model
               
"""原始unet模型权重:unet.hdf5                        模型：unet.get_unet
   三分支模型（长款不一样卷积核）权重：unet1.hdf5    模型：model3.create_model3
   三分支模型（正方形卷积核）权重：unetfangxing.hdf5   模型：model2.create_model2
   三分支后融合权重：unet_change.hdf5                  模型：model1.craete_model1
   
   环绕模型权重：new_model.hdf5                        模型：new_model.new_model

"""
#这个文件包含训练函数和测试函数

path_train='D:/DSB2018_03_18/npyFile/trainIMG.npy'
path_train_mask='D:/DSB2018_03_18/npyFile/trainMASK.npy'
path_test='D:/DSB2018_03_18/npyFile/testIMG.npy'

#模型权重信息文件地址
#file_hdf5='new_model.hdf5'
img_rows = 224
img_cols = 224

smooth = 1.




def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss (y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

#训练函数	
def train():
    print('-'*30)
    print('Loading  train data...')
    print('-'*30)
   
    imgs_train=np.load(path_train).transpose((0,3,1,2))
    imgs_mask_train=np.load(path_train_mask).transpose((0,3,1,2))
    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')

    total=imgs_train.shape[0]
  
    datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    vertical_flip=True,
    horizontal_flip=True)


    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = new_unet(img_rows, img_cols)
   # model.load_weights('new_unet.hdf5')
    model_checkpoint = ModelCheckpoint('new_unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
    callbacks_list = [model_checkpoint]
    model.fit_generator(datagen.flow(imgs_train[:5810], imgs_mask_train[:5810], batch_size=5),
                    steps_per_epoch=5810, epochs=120,verbose=1,validation_data=(imgs_train[5811:],imgs_mask_train[5811:]),callbacks=callbacks_list)
    print('-'*30)
    print('训练完毕')
    print(*'-'*30)
#测试函数
def  test():
    imgs_test=np.load(path_test).transpose((0,3,1,2))

    imgs_test = imgs_test.astype('float32')

    model1 = new_unet(224,224)
    model1.load_weights('new_unet.hdf5')
	
	
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
        ax = fig.add_subplot(221)
        plt.title('yuantu')
        ax.imshow(imgs_test[i,0,:,:],'gray')
		#
        ax = fig.add_subplot(222) 
        plt.title('new_unet')
        ax.imshow(imgs_mask_test1[i,0,:,:],'gray')
        ax = fig.add_subplot(223) 
        plt.title('unet1')
        ax.imshow(imgs_mask_test2[i,0,:,:],'gray')
        ax = fig.add_subplot(224) 
        plt.title('fangxingjuanji')
        ax.imshow(imgs_mask_test3[i,0,:,:],'gray')
        #ax.savefig('C:/Users/server/Desktop/u-net-master/u-net-master/先分后和模型test图像/'+str(i)+'.png')		
        plt.show()
    np.save('imgs_mask_test.npy', imgs_mask_test)

if __name__ == '__main__':
    train()
    #test()
