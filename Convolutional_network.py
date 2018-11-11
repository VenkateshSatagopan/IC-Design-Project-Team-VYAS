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

def seperable_conv2d(x,w,b,k=1):
    x=tf.nn.separable_conv2d(x,w,strides=[1,k,k,1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    return x

def relu(x):
    return tf.nn.relu(x)

def maxpool(x,filter_size=2):
    return tf.nn.max_pool(x,ksize=[1,filter_size,filter_size,1],strides=[1,filter_size,filter_size,1],padding='SAME')

def conv2net_layer(x,data_dict,name=None):

    with tf.variable_scope( name ):
        weights = tf.get_variable( name='weights',
                                   dtype=tf.float32,
                                   initializer=tf.Variable( tf.constant(data_dict[ name ][0],
                                                                         name="weights" ) ) )

        biases = tf.get_variable( name='biases',
                                  dtype=tf.float32,
                                  initializer=tf.Variable( tf.constant(data_dict[ name ][ 0 ],
                                                                        name="biases" ) ) )

        conv_layer_1_output=conv2d(x,weights,biases)
        conv_layer_1_relu=relu(conv_layer_1_output)
        max_pool_1 = maxpool( conv_layer_1_relu, 2 )

        return max_pool_1


def seperable_conv_layer(x,data_dict,name=None):

    with tf.variable_scope( name ):
        weights = tf.get_variable( name='weights',
                                   dtype=tf.float32,
                                   initializer=tf.Variable( tf.constant(data_dict[ name ][0],
                                                                         name="weights" ) ) )

        biases = tf.get_variable( name='biases',
                                  dtype=tf.float32,
                                  initializer=tf.Variable( tf.constant(data_dict[ name ][ 0 ],
                                                                        name="biases" ) ) )

        conv_layer_1_output=seperable_conv2d(x,weights,biases)
        conv_layer_1_relu=relu(conv_layer_1_output)
        max_pool_1 = maxpool( conv_layer_1_relu, 2 )

        return max_pool_1



def conv_layer_layer_3(x,data_dict,name=None):

    with tf.variable_scope( name ):
        weights = tf.get_variable( name='weights',
                                   dtype=tf.float32,
                                   initializer=tf.Variable( tf.constant(data_dict[ name ][0],
                                                                         name="weights" ) ) )

        biases = tf.get_variable( name='biases',
                                  dtype=tf.float32,
                                  initializer=tf.Variable( tf.constant(data_dict[ name ][ 0 ],
                                                                        name="biases" ) ) )

        conv_layer_3_1_output=seperable_conv2d(x,weights,biases)
        conv_layer_3_1_relu=relu(conv_layer_3_1_output)
        conv_layer_3_1_bn = tf.layers.batch_normalization( conv_layer_3_1_relu, axis=-1,
                                                         momentum=0.9,
                                                         epsilon=0.00001,
                                                         center=True,
                                                         scale=True,
                                                         training=True,
                                                         # fused=True
                                                         # data_format='NCHW'
                                                         )

        conv_layer_3_2_output = seperable_conv2d(conv_layer_3_1_bn, weights, biases )
        conv_layer_3_2_relu = relu( conv_layer_3_2_output )
        conv_layer_3_2_bn = tf.layers.batch_normalization( conv_layer_3_2_relu, axis=-1,
                                                           momentum=0.9,
                                                           epsilon=0.00001,
                                                           center=True,
                                                           scale=True,
                                                           training=True,
                                                           # fused=True
                                                           # data_format='NCHW'
                                                           )
        conv_layer_3_3_output = seperable_conv2d( conv_layer_3_2_bn, weights, biases )
        conv_layer_3_3_relu = relu( conv_layer_3_3_output )
        conv_layer_3_3_bn = tf.layers.batch_normalization( conv_layer_3_3_relu, axis=-1,
                                                           momentum=0.9,
                                                           epsilon=0.00001,
                                                           center=True,
                                                           scale=True,
                                                           training=True,
                                                           # fused=True
                                                           # data_format='NCHW'
                                                           )
        max_pool_output = maxpool( conv_layer_3_3_bn, 2 )

        return max_pool_output


