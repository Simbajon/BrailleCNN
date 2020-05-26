# BrailleCNN
A  Simpe Tensorflow/ Keras Implementation of Braille Character Recognition using CNN

File/ Folder Description:

Brail:-
     # This folder consists of subfolders of each alphabet. Each alphabet subfolder contains one corresponding Braille character image.
     # You can add multiple images for generating a bigger Dataset
     
     
Braille_Dataset.zip:-
     # This zip has a folder that contains the images generated using Datagen.py 
     
     #This is the dataset which is used for training
     
     #Image Description:
        Each image is a 28x28 image in BW Scale.
        Each image name consists of the character alphabet and the number of the image
        and the type of data augmentation it went through.
        (i.e whs - width height shift, rot - Rotation, dim - brightness)
        Dataset composition:
        26 characters * 3 Augmentations * 20 different images of different augmentation values (i.e different shift,rotational and      brightness values.)
        
BrailleTrain.py:-
      # The file used for training the CNN Model
      
      #Library used was Keras
      
BrailleTest.py:-
      # The file used for training 
      
      # Pass your own image file path at line 35 for prediction. (Try the a1.jpg and p.jpg file)
      
      
Braille.h5:-
      # PreTrained Model File.
      
HyperBraille.ipynb:-
      # A HyperResnet training Implementation of Braille Recogntion:
      
      # For this purpose KerasTuner was Used
      # The HyperResnet tuner along with Bayesian Optimization was used for getting the best Model
      # The number of training epochs was set to 50 with max_trials for search set to 3
      
      #Results:-
          # The accuracy of the previous Simple CNN Implementation was 88% whereas the Hyper Implementation was 93.4%.
          # This accuracy can be increased by running the Tuner for more trials and by increasing epochs.
          
      
