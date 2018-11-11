from pathlib import Path
import tensorflow as tf
import numpy as np
import sys
import os
from scipy import misc
import glob
 #Getting current directory
current_directory=os.getcwd()
#print(current_directory)
'''#Preparing path for data
data_dir=os.path.join(current_directory+'/Input/chest_xray/')

#data_dir=current_directory+'/input/chest_xray/'

#data_dir=Path(data_dir)

train_dir = os.path.join(data_dir,"train")

valid_dir= os.path.join(data_dir,"val")

test_dir = os.path.join(data_dir,"test")'''


'''def normalize_image( image, axis=None ):
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean( image, axis=axis, keepdims=True )
    std = np.sqrt( ((image - mean) ** 2).mean( axis=axis, keepdims=True ) )
    return (image - mean) / std'''

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total
     # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)
     # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def prepare_train_data(train_dir,tf_record_path):
 #Followed from tutorial https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb
 #num_images=5216
 normal_cases_dir = os.path.join(train_dir,'NORMAL/*.jpeg')

 pneumonia_cases_dir =os.path.join(train_dir,"PNEUMONIA/*.jpeg")



  # Get the list of all the images
 normal_cases = glob.glob(normal_cases_dir)
 pneumonia_cases = glob.glob(pneumonia_cases_dir)
 print( "\nConverting and store in the path: " + tf_record_path )
 num_images=len(normal_cases)+len(pneumonia_cases)

 with tf.python_io.TFRecordWriter(tf_record_path) as writer:
   count = 0
   for img in normal_cases:
     print_progress( count=count, total=num_images - 1 )
     im=misc.imread(img)
     if len(im.shape)==2:
        im = np.dstack( [ im, im, im ] )
     im=misc.imresize(im,(224,224))
     img_bytes=im.tostring()
     label=0
     data = \
       {
           'image': wrap_bytes( img_bytes ),
           'label': wrap_int64( label )
       }
   # Wrap the data as TensorFlow Features.
     feature = tf.train.Features( feature=data )
   # Wrap again as a TensorFlow Example.
     example = tf.train.Example( features=feature )
   # Serialize the data.
     serialized = example.SerializeToString()
   # Write the serialized data to the TFRecords file.
     writer.write( serialized )
     count += 1

   for img in pneumonia_cases:
       print_progress( count=count, total=num_images - 1 )
       im = misc.imread( img )
       if len( im.shape ) == 2:
           im = np.dstack( [ im, im, im ] )
       im = misc.imresize( im, (224, 224) )
       img_bytes = im.tostring()
       label = 1
       data = \
           {
               'image': wrap_bytes( img_bytes ),
               'label': wrap_int64( label )
           }
       # Wrap the data as TensorFlow Features.
       feature = tf.train.Features( feature=data )
       # Wrap again as a TensorFlow Example.
       example = tf.train.Example( features=feature )
       # Serialize the data.
       serialized = example.SerializeToString()
       # Write the serialized data to the TFRecords file.
       writer.write( serialized )
       count += 1
 assert num_images==count
 return count



