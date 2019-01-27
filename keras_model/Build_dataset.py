import cv2
import numpy as np
import os
import argparse
from random import shuffle
from tqdm import tqdm
import scipy
from scipy import misc
import skimage
from skimage.transform import resize
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import  ModelCheckpoint
from keras.layers import Dense , Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import SeparableConv2D,Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout,LeakyReLU
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.initializers import glorot_uniform,glorot_normal
from keras.callbacks import LearningRateScheduler


def get_data(Dir):

# Function to get training data
# Input - train data directory
# Output- X - Input train data array
#         y - Input train data label 
    X = [ ]
    y = [ ]
    for nextDir in os.listdir( Dir ):
        if not nextDir.startswith( '.' ):
            if nextDir in [ 'NORMAL' ]:
                label = 0
            elif nextDir in [ 'PNEUMONIA' ]:
                label = 1
            else:
                label = 2

            temp = Dir + nextDir

            for file in tqdm( os.listdir( temp ) ):
                if not file==".DS_Store":
                 img = misc.imread( temp + '/' + file )
                 if img is not None:
                    img = misc.imresize(img, (112, 112) )
                    if len(img.shape) == 2:
                        img = np.dstack( [ img, img, img ] )
                    img = np.asarray( img )
                    X.append( img )
                    y.append( label )

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def get_data_test( Dir ):
# Function to prepare test data images
# Input - test data directory
# Output- X - test data array
#         y - test data label 
    X = [ ]
    y = [ ]
    for nextDir in os.listdir( Dir ):
        if not nextDir.startswith( '.' ):
            if nextDir in [ 'NORMAL' ]:
                label = 0
            elif nextDir in [ 'PNEUMONIA' ]:
                label = 1
            else:
                label = 2

            temp = Dir + nextDir

            for file in tqdm( os.listdir( temp ) ):
                if not file==".DS_Store":
                 img = misc.imread( temp + '/' + file )
                 if img is not None:
                    img = misc.imresize(img, (112, 112) )
                    if len(img.shape) == 2:
                        img = np.dstack( [ img, img, img ] )
                    img = np.asarray(img)
                    X.append( img )
                    y.append( label )

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


def seaborn_plot(y_train):
 count = y_train.sum(axis = 0)
 sns.countplot(x = count)


def image_generate(X_train):

# Image generator to do augmentation on training dataset

    image_gen_2 = ImageDataGenerator(  # featurewise_center=True,
        # featurewise_std_normalization=True)

        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest" )


    return image_gen_2.fit(X_train)

def run_machine_learning_model(file,file_1):
# Function to run machine-learning model on Rasperry pi
# Input - file - saved model
#         file1- Input chest x-ray file which has to be predicted by model 
#         y_classes[0] - predicted label
#         label - Actual label

      model = load_model(file)
      print(file_1)
      X = [ ]
      y = [ ]
      if "NORMAL" in file_1:
         label=0
      else:
         label=1
      print('The value of label is '+str(label))
      img=misc.imread(file_1)
      if img is not None:
        img = misc.imresize( img, (112, 112))
        if len(img.shape) == 2:
           img = np.dstack( [ img, img, img ] )
           img = np.asarray(img)
      img=np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
      model.compile( optimizer=Adam(), loss='binary_crossentropy', metrics=[ 'accuracy' ])
      a=model.predict(img)
      y_classes = a.argmax( axis=-1 )
      print("Predicted class is "+str(y_classes[0]))
      print("Actual class is "+str(label))
      return y_classes[0],label
