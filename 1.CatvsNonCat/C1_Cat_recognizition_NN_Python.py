#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:11:29 2019

@author: jerome
"""

import numpy as np
import matplotlib.pyplot as plt
import DNN_Util as util
from PIL import Image
    
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(7)
    costs = []                         # keep track of cost
    
    # Parameters initialization
    parameters = util.initialize_parameters(layers_dims)
    
    # Loop
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = util.L_model_forward(X, parameters)
        
        # Compute cost.
        cost = util.compute_cost(AL, Y)
    
        # Backward propagation.
        grads = util.L_model_backward(AL,Y,caches)
 
        # Update parameters.
        parameters = util.update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def main():
    ### IMPORT DATA ###
    train_x_orig, train_y, test_x_orig, test_y, classes = util.load_data('datasets/train_catvnoncat.h5', 'datasets/test_catvnoncat.h5')
    m_train = train_x_orig.shape[0]
    m_test = test_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    
    print("\n DATA \n")
    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))
    
    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    
    print ("train_x's shape after flattening: " + str(train_x.shape))
    print ("test_x's shape after flattening: " + str(test_x.shape))
    
    ### CONSTANTS DEFINING THE MODEL ####
    n_x = 12288     # num_px * num_px * 3
    n_h1 = 20
    n_h2 = 7
    n_h3 = 5
    n_y = 1
    layers_dims = [n_x, n_h1, n_h2, n_h3, n_y] #  4-layer model    
    
    ### MODEL TRAINING AND EVALUATION ####
    print("\n LEARNING \n")
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1000, print_cost = True)
    print("\n MODEL ACCURACY \n")
    pred_train = util.predict(train_x, train_y, parameters)
    pred_test = util.predict(test_x, test_y, parameters)  
    
    ### EXAMPLE ####
    fname = "images/cat2.jpg"
    label_y = [1]
    
    image = np.array(plt.imread(fname))
    image = np.array(Image.fromarray(image).resize(size=(num_px,num_px)))
    #my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
    
    plt.imshow(image)
    
    image = image.reshape((num_px*num_px*3,1))/255
    
    my_predicted_image = util.predict(image, label_y, parameters)    
    print ("y = " + str(np.squeeze(my_predicted_image)) + ", L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    
if __name__== "__main__":
  main()