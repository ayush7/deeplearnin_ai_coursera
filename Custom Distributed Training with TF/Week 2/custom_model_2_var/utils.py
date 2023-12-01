"""
Contains other helper funtions:

loss_func: Mean squared loss function 

"""

import tensorflow as tf 

def loss_func(pred_y, real_y):
    return tf.reduce_mean(tf.square(pred_y-real_y))

