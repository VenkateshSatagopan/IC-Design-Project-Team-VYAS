from scipy import misc
import os
from pathlib import Path
import tensorflow as tf
import numpy as np


#Getting current directory
current_directory=os.getcwd()
print(current_directory)

#Preparing path for data
data_dir=current_directory+'/Input/chest_xray/'

data_dir=Path(data_dir)

# Path to train directory
train_dir = data_dir / 'train'

# Path to validation directory
val_dir = data_dir / 'val'

# Path to test directory
test_dir = data_dir / 'test'


def prepare_train_data(train_dir):
 images=[]
 labels=[]
 normal_cases_dir = train_dir / 'NORMAL'
 pneumonia_cases_dir = train_dir / 'PNEUMONIA'

 # Get the list of all the images
 normal_cases = normal_cases_dir.glob( '*.jpeg' )
 pneumonia_cases = pneumonia_cases_dir.glob( '*.jpeg' )



 for img in normal_cases:
     im=misc.imread(img)
     if len(im.shape)==2:
        im = np.dstack( [ im, im, im ] )
     im = tf.reshape( im, [ im.shape[ 0 ], im.shape[ 1 ], 3 ] )
     im=tf.image.resize_image_with_crop_or_pad(im,224,224)
     norm_image=tf.image.per_image_standardization(im)
     images.append(norm_image)
     labels.append(0)
 for img in pneumonia_cases:
     im=misc.imread(img)
     if len(im.shape)==2:
        im = np.dstack( [ im, im, im ] )
     im = tf.reshape( im, [ im.shape[ 0 ], im.shape[ 1 ], 3 ] )
     im=tf.image.resize_image_with_crop_or_pad(im,224,224)
     norm_image = tf.image.per_image_standardization( im )
     images.append( norm_image )
     labels.append(1)

 images=tf.stack(images)



 return labels,images


images,labels=prepare_train_data(train_dir)

init=tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    res,lab=sess.run([images,labels])
    print(res.shape)
    print(len(lab))





