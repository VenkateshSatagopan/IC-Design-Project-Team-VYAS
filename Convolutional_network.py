from pathlib import Path
import tensorflow as tf
import numpy as np
import sys
import os
from scipy import misc
import glob


def conv2d(x,w,b,k=1):
    # Function used to perform 2D convolution
    x = tf.nn.conv2d(x, w, strides=[1, k, k, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x

def seperable_conv2d(x,w,w1,b,k=1):
    x=tf.nn.separable_conv2d(x,w,w1,strides=[1,k,k,1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    return x

def relu(x):
    return tf.nn.relu(x)

def maxpool(x,filter_size=2):
    return tf.nn.max_pool(x,ksize=[1,filter_size,filter_size,1],strides=[1,filter_size,filter_size,1],padding='SAME')

def conv2net_layer(x,data_dict,name=None,max_pool=False):


        ''' weights = tf.get_variable( name='weights',
                                   dtype=tf.float32,
                                   initializer=tf.Variable( tf.constant(data_dict[name][0],
                                                                         name="weights" ) ) )

        biases = tf.get_variable( name='biases',
                                  dtype=tf.float32,
                                  initializer=tf.Variable( tf.constant(data_dict[ name ][ 1 ],
                                                                        name="biases" ) ) )'''
        weights=data_dict[name][0]
        biases=data_dict[name][1]
        conv_layer_1_output=conv2d(x,weights,biases)
        conv_layer_1_relu=relu(conv_layer_1_output)
        if max_pool:
         conv_layer_1_relu = maxpool(conv_layer_1_relu, 2 )

        return conv_layer_1_relu


def seperable_conv_layer(x,weights_depthwise,weights_pointwise,biases,max_pool=False):

        '''with tf.variable_scope( name ):
        weights = tf.get_variable( name='weights',
                                   dtype=tf.float32,
                                   initializer=tf.Variable( tf.constant(data_dict[ name ][0],
                                                                         name="weights" ) ) )

        biases = tf.get_variable( name='biases',
                                  dtype=tf.float32,
                                  initializer=tf.Variable( tf.constant(data_dict[ name ][ 0 ],
                                                                        name="biases" ) ) )'''


        conv_layer_1_output=seperable_conv2d(x,weights_depthwise,weights_pointwise,biases)
        conv_layer_1_relu=relu(conv_layer_1_output)
        if max_pool:
         conv_layer_1_relu = maxpool( conv_layer_1_relu, 2 )

        return conv_layer_1_relu



def seperable_conv_layer_bn(x,weights_depthwise,weights_pointwise,biases,max_pool=False,is_train=True):



        conv_layer_3_1_output=seperable_conv2d(x,weights_depthwise,weights_pointwise,biases)

        conv_layer_3_1_bn = tf.layers.batch_normalization(conv_layer_3_1_output, axis=-1,
                                                         momentum=0.9,
                                                         epsilon=0.00001,
                                                         center=True,
                                                         scale=True,
                                                         training=is_train
                                                         )
        conv_layer_3_1_relu = relu( conv_layer_3_1_bn )

        if max_pool:
            conv_layer_3_1_relu=maxpool(conv_layer_3_1_relu, 2 )

        return conv_layer_3_1_relu



def fully_connected_layer(x,weights,biases,use_relu=True):
    reshape_input = tf.reshape(x, [-1, weights.get_shape().as_list()[ 0 ]])
    output=tf.matmul(reshape_input, weights)
    output = tf.add(output, biases)
    if use_relu:
        output = tf.nn.relu(output )
    return output
