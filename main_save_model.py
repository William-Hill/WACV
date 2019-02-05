from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from read_input import *
from train import *
from testing import *
from rnn import *
import numpy as np
#import skimage.measure
#State from ground truth, initial position:given, output location feed to input as mask
#(6, 60101, 256, 513)
#h=256; w=513;
h=128
w=257
#large data: 16 sec. small data: 5 sec

#1: Log files
fout_log= open("log.txt","w")
fout_log.write("TEST LOG PRINT\nSOO\n")

def print_ops():
    graph = tf.get_default_graph()
    ops = graph.get_operations()
    print("Total ops:", len(ops))
    for op in ops:
        print(op.name, op.type)

#2: Graph
#Training Parameters
validation_step=10;
learning_rate =0.001 #0.001
X = tf.placeholder("float", [FLAGS.batch_size, None, h,w,channels],name="x") #shape=(24, ?, 256, 513, 6)
Y = tf.placeholder("float", [FLAGS.batch_size, None, h,w,1],name="y") #shape=(24, ?, 256, 513, 1)
timesteps = tf.shape(X)[1]
h=tf.shape(X)[2] #h:256
w=tf.shape(X)[3] #w:513

prediction, last_state = ConvLSTM(X) #shape=(24, ?, 256, 513, 1)
loss_op=tf.losses.mean_pairwise_squared_error(Y,prediction)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#3 :Training
with tf.Session() as sess:
    saver = tf.train.Saver()
    # Initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)
    train_X,train_Y,test_X,test_Y,val_X,val_Y=read_input()
#size of training, test, validation input/outpput
#shape=[number of package, batch_size, timesteps, h, w, channels]
#input channels channel[0]= east-ward wind
#               channels[1]=north-ward wind
#               channel[2]=precipitation
#Training:
#input train_X (104, 24, 10, 128, 257, 3)
#output train_Y (104, 24, 10, 128, 257, 1)
#Test:
#input test_X (28, 24, 10, 128, 257, 3)
#output test_Y (28, 24, 10, 128, 257, 1)
#Validation:
#input val_X (10, 24, 10, 128, 257, 3)
#output val_Y (10, 24, 10, 128, 257, 1)
    print("finished collecting data")
    for ii in range(1000):
        #Soo trouble shooting
        saver.restore(sess, "./my_model.ckpt")
        print("model restored")
        print_ops()
        train(sess,loss_op,train_op,X,Y,train_X,train_Y,val_X,val_Y,prediction, last_state,fout_log)
        name=str(ii)
        test(name,sess,loss_op,train_op,X,Y,test_X,test_Y,prediction,last_state,fout_log)
        if ii%1==0:
             # Save
             save_path = saver.save(sess, "./my_model.ckpt")
             print("Model saved")
fout_log.close();
