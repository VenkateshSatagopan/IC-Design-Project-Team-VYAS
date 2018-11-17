from pathlib import Path
import tensorflow as tf
import numpy as np
import sys
import os
from scipy import misc
import glob
import random
import h5py
import logging
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

#epochs = 1            # Total number of training epochs
#batch_size = 100        # Training batch size
#display_freq = 100      # Frequency of displaying the training results
#learning_rate = 0.001   # The optimization initial learning rate

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
no_epochs=5
learning_rate=0.00005


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

train_dataset = (tf.data.TFRecordDataset(path_tfrecords_train)
           .shuffle(buffer_size=train_images_count,seed=random_number)  # step 1: all the  filenames into the buffer ensures good shuffling
           .repeat(no_epochs)
           .map(read_and_decode, num_parallel_calls=num_parallel_calls)  # step 2

           #.map( get_patches_fn, num_parallel_calls=num_parallel_calls )  # step 3
           #.apply( tf.data.experimental.unbatch())  # unbatch the patches we just produced
           #.apply(tf.contrib.data.unbatch())
           #.shuffle( buffer_size=buffer_size, seed=random_number_1 )  # step 4
           #.repeat(2)
           .batch( batch_size )  # step 5
           .prefetch(1)  # step 6: make sure you always have one batch ready to serve
           )
iterator = train_dataset.make_one_shot_iterator()
image,label = iterator.get_next()


valid_train_dataset = (tf.data.TFRecordDataset(path_tfrecords_valid)
           .shuffle(buffer_size=valid_images_count,seed=random_number )  # step 1: all the  filenames into the buffer ensures good shuffling
           .repeat(no_epochs*500)
           .map(read_and_decode, num_parallel_calls=num_parallel_calls )  # step 2

           #.map( get_patches_fn, num_parallel_calls=num_parallel_calls )  # step 3
           #.apply( tf.data.experimental.unbatch())  # unbatch the patches we just produced
           #.apply(tf.contrib.data.unbatch())
           #.shuffle( buffer_size=buffer_size, seed=random_number_1 )  # step 4
           #.repeat(2)
           .batch(batch_size)  # step 5
           .prefetch(1)  # step 6: make sure you always have one batch ready to serve
           )
iterator_val = valid_train_dataset.make_one_shot_iterator()
image_val,label_val = iterator_val.get_next()


#init=tf.global_variables_initializer()


'''with tf.Session() as sess:
    sess.run(init)
    res,lab=sess.run([image,label])
    print(res.shape)
    print(lab)'''

#np.savez_compressed('/home/venkatesh/Desktop/Lecture_Materials/Advanced_IC_Design/project/vgg16_weights.npz')
#weight_file="vgg16.npy"
#weights = np.load(weight_file)
#data = np.load(weight_file, encoding='latin1').item()




#temp=conv2net_layer(image,data,name='conv1_1',max_pool=False)
#temp1=conv2net_layer(temp,data,name='conv1_2',max_pool=True)
temp=conv2net_layer(image,weights_layer_1['wc1'],biases_layer_1['bc1'],max_pool=False)
temp1=conv2net_layer(temp,weights_layer_2['wc1'],biases_layer_2['bc1'],max_pool=True)
#temp1=conv2net_layer(temp,data,name='conv1_2',max_pool=True)
temp2=seperable_conv_layer(temp1,weights_layer_3['wc1'],weights_layer_3['wc2'],biases_layer_3['bc1'],max_pool=False)
temp3=seperable_conv_layer(temp2,weights_layer_3['wc3'],weights_layer_3['wc4'],biases_layer_3['bc2'],max_pool=True)
temp4=seperable_conv_layer_bn(temp3,weights_layer_4['wc1'],weights_layer_4['wc2'],biases_layer_4['bc1'],max_pool=False,is_train=True)
temp5=seperable_conv_layer_bn(temp4,weights_layer_4['wc3'],weights_layer_4['wc4'],biases_layer_4['bc2'],max_pool=False,is_train=True)
temp6=seperable_conv_layer(temp5,weights_layer_4['wc5'],weights_layer_4['wc6'],biases_layer_4['bc3'],max_pool=True)
temp7=seperable_conv_layer_bn(temp6,weights_layer_5['wc1'],weights_layer_5['wc2'],biases_layer_5['bc1'],max_pool=False,is_train=True)
temp8=seperable_conv_layer_bn(temp7,weights_layer_5['wc3'],weights_layer_5['wc4'],biases_layer_5['bc2'],max_pool=False,is_train=True)
temp9=seperable_conv_layer(temp8,weights_layer_5['wc5'],weights_layer_5['wc6'],biases_layer_5['bc3'],max_pool=True)

# Newly added function##

temp7_1=seperable_conv_layer(temp9,weights_layer_5_1['wc1'],weights_layer_5_1['wc2'],biases_layer_5_1['bc1'],max_pool=True)
#temp8_1=seperable_conv_layer_bn(temp7_1,weights_layer_5_1['wc3'],weights_layer_5_1['wc4'],biases_layer_5_1['bc2'],max_pool=False,is_train=True)
#temp9_1=seperable_conv_layer(temp8_1,weights_layer_5_1['wc5'],weights_layer_5_1['wc6'],biases_layer_5_1['bc3'],max_pool=True)


#Fully connected_layer
temp10=fully_connected_layer(temp7_1,weights_layer_6['wc1'],biases_layer_6['bc1'],use_relu=True)
temp10_drop=tf.nn.dropout(temp10,keep_prob=0.7)
temp11=fully_connected_layer(temp10_drop,weights_layer_6['wc2'],biases_layer_6['bc2'],use_relu=True)
temp11_drop=tf.nn.dropout(temp11,keep_prob=0.5)
output_layer=fully_connected_layer(temp11_drop,weights_layer_6['wc3'],biases_layer_6['bc3'],use_relu=False)
output_layer_transpose=tf.reshape(output_layer,[1,batch_size])
label=tf.reshape(label,[1,batch_size])


#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label,logits=output_layer_transpose),name='loss')

prediction=tf.sigmoid(output_layer_transpose)
predicted_class=tf.greater_equal(prediction,0.5)

# Newly added here
#cross_entropy=tf.nn.weighted_cross_entropy_with_logits(logits=prediction,targets=label,pos_weight=2.7078966)
#cross_entropy=tf.nn.weighted_cross_entropy_with_logits(logits=prediction,targets=label,pos_weight=1.0)
#cross_entropy=tf.nn.weighted_cross_entropy_with_logits(logits=prediction,targets=label,pos_weight=0.49773913043)
#loss=tf.reduce_mean(cross_entropy,name='loss')

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label,logits=output_layer_transpose),name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op')
global_step_layer_1 = tf.train.get_or_create_global_step()
with tf.control_dependencies( tf.get_collection( tf.GraphKeys.UPDATE_OPS ) ):
    train_op_1 = optimizer.minimize(loss, global_step=global_step_layer_1)
correct = tf.equal(predicted_class,tf.equal(label,1))
accuracy = tf.reduce_mean( tf.cast(correct, 'float') )



#validation
#temp_val=conv2net_layer(image_val,data,name='conv1_1',max_pool=False)
#temp1_val=conv2net_layer(temp_val,data,name='conv1_2',max_pool=True)

temp_val=conv2net_layer(image_val,weights_layer_1['wc1'],biases_layer_1['bc1'],max_pool=False)
temp1_val=conv2net_layer(temp_val,weights_layer_2['wc1'],biases_layer_2['bc1'],max_pool=True)
temp2_val=seperable_conv_layer(temp1_val,weights_layer_3['wc1'],weights_layer_3['wc2'],biases_layer_3['bc1'],max_pool=False)
temp3_val=seperable_conv_layer(temp2_val,weights_layer_3['wc3'],weights_layer_3['wc4'],biases_layer_3['bc2'],max_pool=True)
temp4_val=seperable_conv_layer_bn(temp3_val,weights_layer_4['wc1'],weights_layer_4['wc2'],biases_layer_4['bc1'],max_pool=False,is_train=False)
temp5_val=seperable_conv_layer_bn(temp4_val,weights_layer_4['wc3'],weights_layer_4['wc4'],biases_layer_4['bc2'],max_pool=False,is_train=False)
temp6_val=seperable_conv_layer(temp5_val,weights_layer_4['wc5'],weights_layer_4['wc6'],biases_layer_4['bc3'],max_pool=True)
temp7_val=seperable_conv_layer_bn(temp6_val,weights_layer_5['wc1'],weights_layer_5['wc2'],biases_layer_5['bc1'],max_pool=False,is_train=False)
temp8_val=seperable_conv_layer_bn(temp7_val,weights_layer_5['wc3'],weights_layer_5['wc4'],biases_layer_5['bc2'],max_pool=False,is_train=False)
temp9_val=seperable_conv_layer(temp8_val,weights_layer_5['wc5'],weights_layer_5['wc6'],biases_layer_5['bc3'],max_pool=True)
# Newly added function
temp7_val_1=seperable_conv_layer(temp9_val,weights_layer_5_1['wc1'],weights_layer_5_1['wc2'],biases_layer_5_1['bc1'],max_pool=True)
#temp8_val_1=seperable_conv_layer_bn(temp7_val_1,weights_layer_5_1['wc3'],weights_layer_5_1['wc4'],biases_layer_5_1['bc2'],max_pool=False,is_train=False)
#temp9_val_1=seperable_conv_layer(temp8_val_1,weights_layer_5_1['wc5'],weights_layer_5_1['wc6'],biases_layer_5_1['bc3'],max_pool=True)
####



temp10_val=fully_connected_layer(temp7_val_1,weights_layer_6['wc1'],biases_layer_6['bc1'],use_relu=True)
temp10_drop_val=tf.nn.dropout(temp10_val,keep_prob=0.7)
temp11_val=fully_connected_layer(temp10_drop_val,weights_layer_6['wc2'],biases_layer_6['bc2'],use_relu=True)
temp11_drop_val=tf.nn.dropout(temp11_val,keep_prob=0.5)
output_layer_val=fully_connected_layer(temp11_drop_val,weights_layer_6['wc3'],biases_layer_6['bc3'],use_relu=False)
output_layer_transpose_val=tf.reshape(output_layer_val,[1,batch_size])
label_val=tf.reshape(label_val,[1,batch_size])

loss_val = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_val,logits=output_layer_transpose_val),name='loss_val')
prediction_val=tf.sigmoid(output_layer_transpose_val)
predicted_class_val=tf.greater_equal(prediction_val,0.5)
correct_val = tf.equal(predicted_class_val,tf.equal(label_val,1))
accuracy_val = tf.reduce_mean( tf.cast(correct_val, 'float') )
#confusion = tf.confusion_matrix(labels=label_val, predictions=prediction_val, num_classes=2)
best_acc=-1
step=1
best_model_count=1
model_dir_last="trained_model/last_weights"
model_dir_best="trained_model/best_weights"
saver=tf.train.Saver()
last_saver=tf.train.Saver()
best_saver=tf.train.Saver(max_to_keep=1)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #lo,ac=sess.run([accuracy,loss])
    #print(lo)
    for i in range(0,train_images_count*no_epochs,batch_size):
     #acc,los=sess.run([accuracy,loss])
     #
     #if i%32==0:
     if i%320==0:     
      
      val,lab=sess.run([accuracy_val,loss_val])
      print("val_acc = "+str(val))
      print("val_los = " + str(lab) )
      #print("confusion matrix = " + str(confusion_matrix))
      if best_acc<val:
        best_acc=val
        best_save_path = os.path.join(model_dir_best, 'best_weights')
        best_save_path = best_saver.save(sess, best_save_path, global_step=best_model_count)
        logging.info("- Found new best accuracy, saving in {}".format(best_save_path))
        best_model_count+=1
     elif i==step*(train_images_count-batch_size):
         last_save_path = os.path.join(model_dir_last, 'last_weights')
         last_saver.save(sess, last_save_path, global_step=step)
         logging.info("- Epoch completed, model saving in {}".format(last_save_path))
         #print(" Step "+str(step)+" completed")
         
         step=step+1
     else:
         sess.run(train_op_1)
    save_path = saver.save(sess, "trained_model/model")

'''with tf.Session() as sess:
 sess.run(init)
 res,lab=sess.run([output_layer_transpose_val,label_val])
 print(res.shape)
 print(lab.shape)'''

     #print("acc= "+str(acc))

