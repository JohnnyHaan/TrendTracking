#
# Stock prediction model structure design
# 
# Author: JohnnyHaan
# Data: May_17Th_2020
#
# 

import tensorflow as tf


def cnn_models(inputs_, is_training_=True, keep_prob_=0.5, n_classes_=2):
    '''design the cnn model'''

    conv1 = tf.layers.conv1d(inputs=inputs_, filters=32, kernel_size=3, strides=1, padding='valid', activation = tf.nn.leaky_relu)
    
    conv2 = tf.layers.conv1d(inputs=conv1, filters=64, kernel_size=3, strides=1, padding='valid', activation = tf.nn.leaky_relu)

    conv3 = tf.layers.conv1d(inputs=conv2, filters=128, kernel_size=3, strides=1, padding='valid', activation = tf.nn.leaky_relu)    
    conv4 = tf.layers.conv1d(inputs=conv3, filters=256, kernel_size=3, strides=2, padding='valid', activation = tf.nn.leaky_relu) 
    
    conv5 = tf.layers.conv1d(inputs=conv4, filters=256, kernel_size=1, strides=1, padding='valid', activation = tf.nn.leaky_relu)
    norm5 =tf.layers.batch_normalization(conv5, epsilon=1e-5, momentum=0.99, training=is_training_) 
    
    # Flatten and add dropout
    flat = tf.reshape(norm5, (-1, 256))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    
    # Predictions
    logits = tf.layers.dense(flat, n_classes_, name = 'logits')
    output=tf.identity(logits,name='output')
    
    return logits, output

