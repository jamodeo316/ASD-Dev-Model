import numpy as np
import os
from visual_input import *
from NeuralNetwork import *

# Press the green arrow button in the gutter to run the script.
if __name__ == '__main__':

    # Number of units in the vector made from an image, effectively input layer
    # You can alter this to change the size of the layers in the neural network (units in vector = nodes in layer)
    image_vector_size = 400

    # List of image class categories, names MUST match the names of folders holding the images in image directory
    # You can alter the list items to change the image classes on which the neural network is trained then tested
    class_list = ['trucks', 'dogs', 'cats', 'cars']

    # This is the path to the training image set for the local machine. UPDATE as needed
    train_image_directory = os.getcwd() + "/image_set/train_subset"

    # Dictionary of image class label (key)-list of image vector (value) pairs for training the neural network
    train_image_dict = image2matrix(train_image_directory, class_list, image_vector_size)

    # This is the path to the testing image set for the local machine. UPDATE as needed
    test_image_directory = os.getcwd() + "/image_set/test_subset"

    # Dictionary of image class label (key)-list of image vector (value) pairs for testing the neural network
    test_image_dict = image2matrix(test_image_directory, class_list, image_vector_size)

    # Dictionary of image class label (key)-encoded class label (value) pairs for comparing neural network output
    encoded_label_dict = encode_image_labels(train_image_dict)

    # This is the neural network (an instance of the NeuralNetwork class)
    # You can alter the 'hidden_layer_number', that is the number of layers between the input and output layers
    # You can alter the 'ccl5_delta' for fold differences above/below normal (e.g., 1.5 fold above, enter 0.4)
    # You can alter the 'wnt_delta' for fold differences above/below normal (e.g., 1.5 fold below, enter -0.5)
    net = NeuralNetwork(class_list, image_vector_size, hidden_layer_number=4, ccl5_delta=0, wnt_delta=0)

    # This is the command to train the neural network
    # You can alter the learning rate ('learn_r') to make the size of weight updates bigger or smaller
    # You can alter 'sessions' to change how many times the neural network runs the full image set during training
    net.train(train_image_dict, encoded_label_dict, learn_r=0.1, sessions=100)

    # This is the command to test the neural network
    # You can set 'use_release_maps' to 'True'/'False' for biological dynamic weighting of signals or fixed weighting
    # Note: Because fixed weighting is (by definition) less variable, it may lead to better performance
    net.test(test_image_dict, encoded_label_dict, use_release_maps=True)



