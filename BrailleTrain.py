from matplotlib import pyplot as plt 
from PIL import Image
from keras.preprocessing.image import load_img
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf

x=[]
y=[]
y1=[]
num=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
alp=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
rootDir = "./Brail"
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        im=load_img(dirName+"//"+fname)
        im1=np.asarray(im)
        x.append(im1)
        y.append(fname[0])
x_train=np.asarray(x)
y_train=np.asarray(y)
for i in y_train:
    for alpha in alp:
        if(i==alpha):
            i=num[alp.index(alpha)]
            y1.append(i)
y_train=np.asarray(y1)
x_train = x_train.astype('float32')
x_train /= 255
input_shape = (28, 28, 3)

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) 
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(27,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=30)
model.save('Braille.h5') 


