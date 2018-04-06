"""
三分支后融合在一起的模型"""







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
	
	
    conv5_1 = Conv2D(512, 5, activation='relu', padding='same')(pool4_1)
    conv5_1 = Conv2D(512, 5, activation='relu', padding='same')(conv5_1)
	

	
    
    conv5_2 = Conv2D(512, 7, activation='relu', padding='same')(pool4_2)
    conv5_2 = Conv2D(512, 7, activation='relu', padding='same')(conv5_2)

    conv5=merge([conv5_0,conv5_1,conv5_2],mode='concat', concat_axis=1)
	
    


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