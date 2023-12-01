"""
Create a Helper funtion for plotting sctter plots
Inputs: inputs, true_values, predicted_values

"""

import matplotlib.pyplot as plt 

def plotit(inputs, outputs, predicted):
    #compares real and predicted values
    real = plt.scatter(inputs, outputs, color='r')
    pred = plt.scatter(inputs, predicted, color='b')
    plt.legend([real,pred],['Real Data', 'Predicted Data'])
    plt.show()


