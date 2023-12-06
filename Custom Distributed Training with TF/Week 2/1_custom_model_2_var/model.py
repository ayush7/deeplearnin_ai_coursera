"""Create a simpe model for demonstration purposes"""

import tensorflow as tf


class Model(object):
    def __init__(self):
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(4.0)

    def __call__(self, x):
        return self.w * x + self.b 
    