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

test_images_count=prepare_train_data(test_dir,path_tfrecords_test,is_train=False)

weights_layer_1={
'wc1': tf.Variable( tf.random_normal( shape=[ 3, 3, 3, 64 ], stddev=0.01, mean=0.0 ), tf.float32 )
}


biases_layer_1 = {
    'bc1': tf.Variable( tf.random_normal( shape=[ 64 ], stddev=0.01, mean=0.0 ), tf.float32 )
    
    }

weights_layer_2 = {
'wc1': tf.Variable( tf.random_normal( shape=[ 3, 3, 64, 64 ], stddev=0.01, mean=0.0 ), tf.float32 )
}


biases_layer_2 = {
    'bc1': tf.Variable( tf.random_normal( shape=[ 64 ], stddev=0.01, mean=0.0 ), tf.float32 )
    
    }

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


weights_layer_5_1 = {
    'wc1': tf.Variable( tf.random_normal( shape=[ 3, 3, 512, 1 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    'wc2': tf.Variable( tf.random_normal( shape=[ 1, 1, 512, 512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    #'wc3': tf.Variable( tf.random_normal( shape=[ 3, 3, 512, 1 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    #'wc4': tf.Variable( tf.random_normal( shape=[ 1, 1, 512,512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    #'wc5': tf.Variable( tf.random_normal( shape=[ 3, 3, 512,1 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    #'wc6': tf.Variable( tf.random_normal( shape=[ 1, 1, 512,512 ], stddev=0.01, mean=0.0 ), tf.float32 )
    }

biases_layer_5_1 = {
    'bc1': tf.Variable( tf.random_normal( shape=[ 512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    #'bc2': tf.Variable( tf.random_normal( shape=[ 512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    #'bc3': tf.Variable( tf.random_normal( shape=[ 512 ], stddev=0.01, mean=0.0 ), tf.float32 ),
    }

weights_layer_6={
    'wc1': tf.Variable( tf.random_normal( shape=[7*7*512,1024], stddev=0.01, mean=0.0 ), tf.float32 ),
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
batch_size = 8
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



test_train_dataset = (tf.data.TFRecordDataset(path_tfrecords_test)
           #.shuffle(buffer_size=test_images_count,seed=random_number )  # step 1: all the  filenames into the buffer ensures good shuffling
           #.repeat(no_epochs*500)
           .map(read_and_decode, num_parallel_calls=num_parallel_calls )  # step 2
           
           #.map( get_patches_fn, num_parallel_calls=num_parallel_calls )  # step 3
           #.apply( tf.data.experimental.unbatch())  # unbatch the patches we just produced
           #.apply(tf.contrib.data.unbatch())
           #.shuffle( buffer_size=buffer_size, seed=random_number_1 )  # step 4
           #.repeat(2)
           .batch(batch_size)  # step 5
           .prefetch(1)  # step 6: make sure you always have one batch ready to serve
           )
iterator_test = test_train_dataset.make_one_shot_iterator()
image_test,label_test = iterator_test.get_next()





#Testing
temp_test=conv2net_layer(image_test,weights_layer_1['wc1'],biases_layer_1['bc1'],max_pool=False)
temp1_test=conv2net_layer(temp_test,weights_layer_2['wc1'],biases_layer_2['bc1'],max_pool=True)
temp2_test=seperable_conv_layer(temp1_test,weights_layer_3['wc1'],weights_layer_3['wc2'],biases_layer_3['bc1'],max_pool=False)
temp3_test=seperable_conv_layer(temp2_test,weights_layer_3['wc3'],weights_layer_3['wc4'],biases_layer_3['bc2'],max_pool=True)
temp4_test=seperable_conv_layer_bn(temp3_test,weights_layer_4['wc1'],weights_layer_4['wc2'],biases_layer_4['bc1'],max_pool=False,is_train=False)
temp5_test=seperable_conv_layer_bn(temp4_test,weights_layer_4['wc3'],weights_layer_4['wc4'],biases_layer_4['bc2'],max_pool=False,is_train=False)
temp6_test=seperable_conv_layer(temp5_test,weights_layer_4['wc5'],weights_layer_4['wc6'],biases_layer_4['bc3'],max_pool=True)
temp7_test=seperable_conv_layer_bn(temp6_test,weights_layer_5['wc1'],weights_layer_5['wc2'],biases_layer_5['bc1'],max_pool=False,is_train=False)
temp8_test=seperable_conv_layer_bn(temp7_test,weights_layer_5['wc3'],weights_layer_5['wc4'],biases_layer_5['bc2'],max_pool=False,is_train=False)
temp9_test=seperable_conv_layer(temp8_test,weights_layer_5['wc5'],weights_layer_5['wc6'],biases_layer_5['bc3'],max_pool=True)

# Newly added function
temp7_test_1=seperable_conv_layer(temp9_test,weights_layer_5_1['wc1'],weights_layer_5_1['wc2'],biases_layer_5_1['bc1'],max_pool=True)
temp10_test=fully_connected_layer(temp7_test_1,weights_layer_6['wc1'],biases_layer_6['bc1'],use_relu=True)
temp10_drop_test=tf.nn.dropout(temp10_test,keep_prob=0.7)
temp11_test=fully_connected_layer(temp10_drop_test,weights_layer_6['wc2'],biases_layer_6['bc2'],use_relu=True)
temp11_drop_test=tf.nn.dropout(temp11_test,keep_prob=0.5)
output_layer_test=fully_connected_layer(temp11_drop_test,weights_layer_6['wc3'],biases_layer_6['bc3'],use_relu=False)
output_layer_transpose_test=tf.reshape(output_layer_test,[1,batch_size])
label_test=tf.reshape(label_test,[1,batch_size])

loss_test = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_test,logits=output_layer_transpose_test),name='loss_test')
prediction_test=tf.sigmoid(output_layer_transpose_test)
predicted_class_test=tf.greater_equal(prediction_test,0.5)
correct_test = tf.equal(predicted_class_test,tf.equal(label_test,1))
accuracy_test = tf.reduce_mean( tf.cast(correct_test, 'float') )

saver=tf.train.Saver()
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,"trained_model/model")
    acc=[]
    for i in range(0,test_images_count*no_epochs,batch_size):
      val=sess.run([accuracy_test])
      acc.append(val)
    acc=tf.stack(acc)
    acc_avg=tf.reduce_mean(acc)
    acc_final=sess.run(acc_avg)
    print("Avg accuracy of the model is  "+str(acc_final))
'''with tf.Session() as sess:
   sess.run(init)
   res=sess.run(output_layer_transpose_test)
   print(res.shape)'''

    
      
     
