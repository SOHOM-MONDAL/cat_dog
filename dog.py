import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

train_ds = keras.utils.image_dataset_from_directory(
      directory =  'dataset\\dogcat\\train' ,
      labels= 'inferred',
      label_mode = 'int',
      batch_size= 32,  
     image_size=(256, 256)

)
validation_ds = keras.utils.image_dataset_from_directory(
    directory =  'dataset\\dogcat\\validation' ,
    labels= 'inferred',
    label_mode = 'int',
    batch_size= 32,
    image_size =  (256, 256)
)

def process(image,label) :
    image = tf.cast(image/255. ,tf.float32)
    return image,label 

train_ds = train_ds.map(process)
validation_ds= validation_ds.map(process)

#creat cnn model
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding = 'valid', activation = 'relu', input_shape= (256,256,3)))
model.add(MaxPooling2D(pool_size = (2,2),strides=2 ,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding = 'valid', activation = 'relu', input_shape= (256,256,3)))
model.add(MaxPooling2D(pool_size = (2,2),strides=2 ,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding = 'valid', activation = 'relu', input_shape= (256,256,3)))
model.add(MaxPooling2D(pool_size = (2,2),strides=2 ,padding='valid'))           

model.add(Flatten())


model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss= 'binary_crossentropy',metrics= ['accuracy'])

history = model.fit(train_ds,epochs= 10, validation_data = validation_ds)



















