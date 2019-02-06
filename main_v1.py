"""Main module for running ESGF Hurricane Deep Learning Model."""
from __future__ import print_function
import os
import logging
import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from read_input import *
from train import *
from testing import *
from rnn import *
import numpy as np
# State from ground truth, initial position:given, output location feed to input as mask
# (6, 60101, 128, 257)

# Comes from the shape of the data, significant
HEIGHT = 128
WIDTH = 257

# Sets logging level: INFO and WARNING messages are not printed; only error printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Controls visibility of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def check_for_gpu():
    if tf.test.gpu_device_name():
        print('GPU found')
        print("Gpu name:", tf.test.gpu_device_name())
    else:
        print("No GPU found")



#1: Log files
fout_log= open("log.txt","w")
fout_log.write("TEST LOG PRINT\nSOO\n")

#This is the model
#2: Graph
#Training Parameters
#validation_step=10;
learning_rate =0.001
#placeholders; batch_size defined in function.py, channels are dimensions of the input data aka climate variable
X = tf.placeholder("float", [FLAGS.batch_size, None, HEIGHT, WIDTH, channels]) #shape=(24, ?, 128, 257, 3)
Y = tf.placeholder("float", [FLAGS.batch_size, None, HEIGHT, WIDTH, 1]) #shape=(24, ?, 128, 257, 1)

# TODO: are these variables needed, do not look like they are used after defining
timesteps = tf.shape(X)[1]
HEIGHT = tf.shape(X)[2]
WIDTH = tf.shape(X)[3]

prediction, last_state = ConvLSTM(X) #shape=(24, ?, 256, 513, 1)
#minimize the loss between ground truth and prediction
loss_op=tf.losses.mean_pairwise_squared_error(Y,prediction)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#3 :Training
with tf.Session() as sess:
    # Initialize all variables
    saver = tf.train.Saver()
    # TODO: are timesteps, HEIGHT, WIDTH used with this global_variables_initializer?
    init = tf.global_variables_initializer()
    sess.run(init)
    train_X,train_Y,test_X,test_Y,val_X,val_Y=read_input()
    print("finished collecting data")
    for ii in range(100):
        train(sess,loss_op,train_op,X,Y,train_X,train_Y,val_X,val_Y,prediction, last_state,fout_log)
        name=str(ii)
        test(name,sess,loss_op,train_op,X,Y,test_X,test_Y,prediction,last_state,fout_log)
fout_log.close();
