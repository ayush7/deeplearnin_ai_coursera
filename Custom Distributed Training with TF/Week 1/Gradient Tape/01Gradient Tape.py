import numpy as np 
import tensorflow as tf 
from tensorflow import GradientTape
import random


# Learning Rate
LEARNING_RATE = 0.001

# Training Data
x_train = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_train = np.array([-3.0, -1.0,1.0, 3.0, 5.0, 7.0], dtype=float)

# Trainable Variables
w = tf.Variable(random.random(), trainable=True)
b = tf.Variable(random.random(), trainable=True)

# Gradiant through iterations
grad_list = []


# Loss Function
def simple_loss(real_y, pred_y):
    return tf.abs(pred_y - real_y)


# Training Function

def fit_data(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        # Make Prediction
        pred_y = w * real_x + b 
        # Calculate Loss
        reg_loss = simple_loss(real_y, pred_y)
    
    # Calculate Gradient
    w_gradient = tape.gradient(reg_loss, w)
    b_gradient = tape.gradient(reg_loss, b)

    grad_list.append(tape.gradient(reg_loss,[w,b]))


    # Update Variable
    w.assign_sub(w_gradient*LEARNING_RATE)
    b.assign_sub(b_gradient*LEARNING_RATE)



for _ in range(5000):
    fit_data(x_train, y_train)

print(f'y = {w.numpy()}x + {b.numpy()}')