import os
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from PIL import Image 

rootDir = "./Brail"
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        img = load_img(dirName+"\\"+fname)
        newsize = (28, 28) 
        img= img.resize(newsize) 
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        shift=0.2
        datagen1 = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
        datagen2 = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(samples, batch_size=1)
        it1 = datagen1.flow(samples, batch_size=1)
        it2 = datagen2.flow(samples, batch_size=1)
        for i in range(20):
            batch = it.next()
            batch1 = it1.next()
            batch2 = it2.next()
            image = batch[0].astype('uint8')
            image1 = batch1[0].astype('uint8')
            image2 = batch2[0].astype('uint8')
            ima= Image.fromarray(image, 'RGB')
            ima.save("Braille_Dataset/"+fname+str(i)+"dim"+".jpg")
            ima1= Image.fromarray(image1, 'RGB')
            ima1.save("Braille_Dataset/"+fname+str(i)+"whs"+".jpg")
            ima2= Image.fromarray(image2, 'RGB')
            ima2.save("Braille_Dataset/"+fname+str(i)+"rot"+".jpg")
        
        
