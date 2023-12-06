
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

from model import Model
from plotter import plotit
from utils import loss_func

# Define Model
model = Model()

# Define your Variables

TRUE_w = 3.0
TRUE_b = 2.0
NUM_EXAMPLES  = 1000
LEARNING_RATE = 0.1
EPOCHS = 25

# Training Data [generated]
xs = tf.random.normal(shape=[NUM_EXAMPLES])
ys = (xs*TRUE_w) + TRUE_b

 
#    Now from the given generated training data we will be
#    extracting the true values of W and B using training    


# Plot the data once beforehand 

plotit(xs, ys, model(xs))
print('Current Loss: ', loss_func(model(xs), ys).numpy())


"""
Define the training function
"""
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_func(model(inputs), outputs)
    # differentials
    dw, db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate*dw)
    model.b.assign_sub(learning_rate*db)

    return current_loss


# Define the training loop
model = Model()
epochs = range(EPOCHS)

losses = []
w_list, b_list = [],[]

for epoch in epochs:

    w_list.append(model.w.numpy())
    b_list.append(model.b.numpy())
    current_loss = train(model, xs, ys, learning_rate=LEARNING_RATE)
    losses.append(current_loss)
    print(f'Epoch: {epoch}  w: {w_list[-1]}  b: {b_list[-1]}   Loss: {current_loss}')
    



plt.plot(epochs, w_list, 'r',
         epochs, b_list, 'b')

plt.plot([TRUE_w]*len(epochs),'r--',
         [TRUE_b]*len(epochs), 'b--')
plt.legend(['w','b','True w', 'True b'])
plt.show()