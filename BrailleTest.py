import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image 
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import load_img
from keras.utils import plot_model
import os
import cv2

x=[]
y=[]
y1=[]
num=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
alp=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
rootDir ="./Braille_Dataset"
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


im=load_img("a1.jpg")
newsize = (28, 28) 
im = im.resize(newsize) 
im2=np.asarray(im)
im2 = im2.astype('float32')
im2 /= 255

new_model = tf.keras.models.load_model('Braille.h5')
new_model.summary()
print(im2.shape)
print("=======================================================")
pred=new_model.predict(im2.reshape(1,28,28,3))
print("The predicted Braille Character is:")
print(alp[pred.argmax()-1])

print("=======================================================")
loss, acc = new_model.evaluate(x_train, y_train, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
img=cv2.imread("a1.jpg")
cv2.imshow("Image",img)
cv2.waitKey()
