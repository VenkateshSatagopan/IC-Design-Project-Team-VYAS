from pathlib import Path
import tensorflow as tf
import numpy as np
import sys
import os
from scipy import misc
import glob
import random
import h5py
from Build_dataset import *
from Convolutional_network import *

current_directory=os.getcwd()

# Preparing path for data
data_dir=os.path.join(current_directory+'/Input/chest_xray/')

train_dir = os.path.join(data_dir,"train")

valid_dir= os.path.join(data_dir,"val")

test_dir = os.path.join(data_dir,"test")


 # Define path for Tensorflow record
path_tfrecords_train = os.path.join(current_directory,"tf-records/train.tfrecords")
path_tfrecords_valid = os.path.join(current_directory,"tf-records/valid.tfrecords")
path_tfrecords_test = os.path.join(current_directory,"tf-records/test.tfrecords")

#train_images_count=prepare_train_data(train_dir,path_tfrecords_train)
valid_images_count=prepare_train_data(valid_dir,path_tfrecords_valid)
train_images_count=5216
#test_images_count=prepare_train_data(test_dir,path_tfrecords_test)

#print("\n No of train images "+str(train_images_count))
print("\n No of valid images "+str(valid_images_count))
#print("\n No of test images "+str(test_images_count))


# No of parameters
num_patches = 100  # number of patches to extract from each image
patch_size = 50  # size of the patches
buffer_size = 100000# shuffle patches from 50 different big images
num_parallel_calls = 4  # number of threads
batch_size = 32
learning_rate=0.00005
no_epochs=80


def read_and_decode(tf_record_file):
    # Function to read the tensorflow record and return image suitable for patching
    #  Input: tf_record_file - tf record file in which image can be extracted
    #  Output: Image,label

    features = {
        'image': tf.FixedLenFeature( [ ], tf.string ),
        'label': tf.FixedLenFeature( [ ], tf.int64 )

    }
    parsed = tf.parse_single_example(tf_record_file,features)
    image_raw = parsed[ 'image' ]
    image = tf.decode_raw(image_raw, tf.uint8)
    image_shape = tf.stack([ 224, 224, -1 ])
    image = tf.reshape(image, image_shape )
    image = image[ :, :, :3 ]
    image = tf.cast(image, tf.float32)
    image=tf.image.per_image_standardization(image)
    # Get the label associated with the image.
    label = parsed[ 'label' ]
    return image,label

random_number=random.randint(0,10000)

train_dataset = (tf.data.TFRecordDataset(path_tfrecords_train)
           .shuffle(buffer_size=train_images_count,seed=random_number )  # step 1: all the  filenames into the buffer ensures good shuffling
           #.repeat(no_epochs)
           .map(read_and_decode, num_parallel_calls=num_parallel_calls )  # step 2

           #.map( get_patches_fn, num_parallel_calls=num_parallel_calls )  # step 3
           #.apply( tf.data.experimental.unbatch())  # unbatch the patches we just produced
           #.apply(tf.contrib.data.unbatch())
           #.shuffle( buffer_size=buffer_size, seed=random_number_1 )  # step 4
           #.repeat(5)
           .batch( batch_size )  # step 5
           .prefetch(1)  # step 6: make sure you always have one batch ready to serve
           )
iterator = train_dataset.make_one_shot_iterator()
image,label = iterator.get_next()


init=tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    res,lab=sess.run([image,label])
    print(res.shape)
    print(lab)
temp=[]
#np.savez_compressed('/home/venkatesh/Desktop/Lecture_Materials/Advanced_IC_Design/project/vgg16_weights.npz')
weight_file="vgg16.npy"
#weights = np.load(weight_file)
data = np.load(weight_file, encoding='latin1').item()


#print(data["conv2_2"]['weights'])



print("npy file loaded")
#np.load('/home/venkatesh/Desktop/Lecture_Materials/Advanced_IC_Design/project/vgg16_weights.npz')


