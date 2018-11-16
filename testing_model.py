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


device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu

if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
config.gpu_options.allow_growth = True

current_directory=os.getcwd()


# Preparing path for data
data_dir=os.path.join(current_directory+'/Input/chest_xray/')


test_dir = os.path.join(data_dir,"test")


path_tfrecords_test = os.path.join(current_directory,"tf-records/test.tfrecords")

weights_layer_3 = {
    'wc1': tf.Variable( tf.random_normal( shape=[ 3, 3, 64, 1 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc2': tf.Variable( tf.random_normal( shape=[ 1, 1, 64, 128 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc3': tf.Variable( tf.random_normal( shape=[ 3, 3, 128, 1 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc4': tf.Variable( tf.random_normal( shape=[ 1, 1, 128,128 ], stddev=0.01, mean=0.0 ), tf.float32 )
    }

biases_layer_3 = {
    'bc1': tf.Variable( tf.random_normal( shape=[ 128 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'bc2': tf.Variable( tf.random_normal( shape=[ 128 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    }

weights_layer_4 = {
    'wc1': tf.Variable( tf.random_normal( shape=[ 3, 3, 128, 1 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc2': tf.Variable( tf.random_normal( shape=[ 1, 1, 128, 256 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc3': tf.Variable( tf.random_normal( shape=[ 3, 3, 256, 1 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc4': tf.Variable( tf.random_normal( shape=[ 1, 1, 256,256 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc5': tf.Variable( tf.random_normal( shape=[ 3, 3, 256,1 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc6': tf.Variable( tf.random_normal( shape=[ 1, 1, 256,256 ], stddev=0.01, mean=0.0 ), tf.float32 )
    }

biases_layer_4 = {
    'bc1': tf.Variable( tf.random_normal( shape=[ 256 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'bc2': tf.Variable( tf.random_normal( shape=[ 256 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'bc3': tf.Variable( tf.random_normal( shape=[ 256 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    }

weights_layer_5 = {
    'wc1': tf.Variable( tf.random_normal( shape=[ 3, 3, 256, 1 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc2': tf.Variable( tf.random_normal( shape=[ 1, 1, 256, 512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc3': tf.Variable( tf.random_normal( shape=[ 3, 3, 512, 1 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc4': tf.Variable( tf.random_normal( shape=[ 1, 1, 512,512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc5': tf.Variable( tf.random_normal( shape=[ 3, 3, 512,1 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc6': tf.Variable( tf.random_normal( shape=[ 1, 1, 512,512 ], stddev=0.01, mean=0.0 ), tf.float32 )
    }

biases_layer_5 = {
    'bc1': tf.Variable( tf.random_normal( shape=[ 512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'bc2': tf.Variable( tf.random_normal( shape=[ 512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'bc3': tf.Variable( tf.random_normal( shape=[ 512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    }

weights_layer_6={
    'wc1': tf.Variable( tf.random_normal( shape=[14*14*512,1024], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc2': tf.Variable( tf.random_normal( shape=[1024,512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc3': tf.Variable( tf.random_normal( shape=[ 512,1], stddev=0.01, mean=0.0 ), tf.float32 )

}

biases_layer_6 = {
    'bc1': tf.Variable( tf.random_normal( shape=[ 1024 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'bc2': tf.Variable( tf.random_normal( shape=[ 512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'bc3': tf.Variable( tf.random_normal( shape=[ 1 ], stddev=0.01, mean=0.0 ), tf.float32 )
    }

# No of parameters
num_parallel_calls = 4  # number of threads
batch_size = 16
no_epochs=1
learning_rate=0.0001


def read_and_decode(tf_record_file):
    # Function to read the tensorflow record
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
    label=tf.cast(label,tf.float32)
    return image,label

random_number=random.randint(0,10000)



valid_train_dataset = (tf.data.TFRecordDataset(path_tfrecords_valid)
           .shuffle(buffer_size=valid_images_count,seed=random_number )  # step 1: all the  filenames into the buffer ensures good shuffling
           .repeat(no_epochs*500)
           .map(read_and_decode, num_parallel_calls=num_parallel_calls )  # step 2

           #.map( get_patches_fn, num_parallel_calls=num_parallel_calls )  # step 3
           #.apply( tf.data.experimental.unbatch())  # unbatch the patches we just produced
           #.apply(tf.contrib.data.unbatch())
           #.shuffle( buffer_size=buffer_size, seed=random_number_1 )  # step 4
           #.repeat(2)
           .batch(valid_images_count)  # step 5
           .prefetch(1)  # step 6: make sure you always have one batch ready to serve
           )
iterator_val = valid_train_dataset.make_one_shot_iterator()
image_val,label_val = iterator_val.get_next()


weight_file="vgg16.npy"
#weights = np.load(weight_file)
data = np.load(weight_file, encoding='latin1').item()


#validation
temp_val=conv2net_layer(image_val,data,name='conv1_1',max_pool=False)
temp1_val=conv2net_layer(temp_val,data,name='conv1_2',max_pool=True)
temp2_val=seperable_conv_layer(temp1_val,weights_layer_3['wc1'],weights_layer_3['wc2'],biases_layer_3['bc1'],max_pool=False)
temp3_val=seperable_conv_layer(temp2_val,weights_layer_3['wc3'],weights_layer_3['wc4'],biases_layer_3['bc2'],max_pool=True)
temp4_val=seperable_conv_layer_bn(temp3_val,weights_layer_4['wc1'],weights_layer_4['wc2'],biases_layer_4['bc1'],max_pool=False,is_train=False)
temp5_val=seperable_conv_layer_bn(temp4_val,weights_layer_4['wc3'],weights_layer_4['wc4'],biases_layer_4['bc2'],max_pool=False,is_train=False)
temp6_val=seperable_conv_layer(temp5_val,weights_layer_4['wc5'],weights_layer_4['wc6'],biases_layer_4['bc3'],max_pool=True)
temp7_val=seperable_conv_layer_bn(temp6_val,weights_layer_5['wc1'],weights_layer_5['wc2'],biases_layer_5['bc1'],max_pool=False,is_train=False)
temp8_val=seperable_conv_layer_bn(temp7_val,weights_layer_5['wc3'],weights_layer_5['wc4'],biases_layer_5['bc2'],max_pool=False,is_train=False)
temp9_val=seperable_conv_layer(temp8_val,weights_layer_5['wc5'],weights_layer_5['wc6'],biases_layer_5['bc3'],max_pool=True)
temp10_val=fully_connected_layer(temp9_val,weights_layer_6['wc1'],biases_layer_6['bc1'],use_relu=True)
temp10_drop_val=tf.nn.dropout(temp10_val,keep_prob=0.7)
temp11_val=fully_connected_layer(temp10_drop_val,weights_layer_6['wc2'],biases_layer_6['bc2'],use_relu=True)
temp11_drop_val=tf.nn.dropout(temp11_val,keep_prob=0.5)
output_layer_val=fully_connected_layer(temp11_drop_val,weights_layer_6['wc3'],biases_layer_6['bc3'],use_relu=False)
output_layer_transpose_val=tf.reshape(output_layer_val,[1,16])
label_val=tf.reshape(label_val,[1,16])

loss_val = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_val,logits=output_layer_transpose_val),name='loss_val')
prediction_val=tf.sigmoid(output_layer_transpose_val)
predicted_class_val=tf.greater_equal(prediction_val,0.5)
correct_val = tf.equal(predicted_class,tf.equal(label_val,1))
accuracy_val = tf.reduce_mean( tf.cast(correct, 'float') )

