"""
(This is a 'doc string' specified by 3 quotes at the beginning and end. This
allows you to write comments to other programmers that the computer just ignores.)

These lines below are python packages/libraries, bunches of code that other people
have written that they have made publicly available. You will see how we use these
later on, for now this command 'import' tells the computer: we want to use the
listed packages and their files in our project so keep them handy.
"""
import cv2
import os
import numpy as np


# (This is a comment, specified by a hashtag, it also lets you write comments :)
# that the computer ignores. These are best for one-liners though As you can see,
# every new line must start with its own hashtag.

"""
Below is a collection of functions that we will call in out main.py file to help us
do visual processing in our model. When a line of code starts with 'def', it is a 
function. After 'def' comes the name of the function, which is what we will use to call
it and its associated instructions in main.py. Look up 'functions python' online. 
"""


def image2matrix(image_directory, class_list, vector_size):
    # The words inside the () are called 'parameters'. Look up 'python parameters' and 'arguments'.
    """
    Under every function, it is good to write a description of what it does.
    This function collects images of the specified image class from the image dir and
    converts them to a matrix of red,green,blue values (one for each image pixel).

    Parameters
    ----------
    image_directory
    class_list
    vector_size

    Returns
    -------
    image_dict
    """

    image_dict = {}  # Setting the 'variable' equal to an empty 'dictionary'. Look up words in '' with 'python'.

    for image_class in class_list:
        # Look up 'for loop'

        class_path = image_directory + "/" + image_class  # Making path to images of specific class. Look up 'strings'
        image_list = os.listdir(class_path)  # 'os' is a built-in python object. Look it up and 'methods'

        vector_list = []  # Look up 'list'
        for image in image_list:
            # 'for loop' again

            image_path = os.path.join(class_path, image)  # os 'method' sticks together directory and image paths
            image_matrix = cv2.imread(image_path)  # Has computer follow path made above to get image
            # Using the 'imread' method of the 'cv2' package we imported to turn image into matrix of numbers

            gray_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)  # Converts b,g,r values to grayscale values
            resized_matrix = cv2.resize(gray_matrix, (int(vector_size**0.5), int(vector_size**0.5)))  # Resizes matrix
            normalized_matrix = resized_matrix.astype(np.float32) / 255.0
            # Turning values into percentage of 255 so all of them are comparable, important of machine learning

            image_vector = normalized_matrix.flatten().reshape(-1, 1)  # Flattens matrix and shapes it as 2500x1
            vector_list.append(image_vector)  # Using the 'append' method to add the image vector to the vector list

        image_dict[image_class] = vector_list
        # Adding 'key:value pair' to image dict with key = image class and value = vector list. Look up words in ''.

    return image_dict  # 'return' statement is what function gives back after running. Look up


def encode_image_labels(image_dict):
    """
    This function turns string labels for image class (e.g., 'dog') into a
    numpy array that encodes the image label in a series of 0s and 1s.

    Parameters
    ----------
    image_dict

    Returns
    -------
    encoded_label_dict
    """

    encoded_label_dict = {}  # Creating empty 'dictionary', look up
    class_count = len(image_dict.keys())  # Counting classes in image dict passed into function and saving in variable

    for index in range(class_count):
        # 'for loop' again, 'range' creates a list of nums starting at 0 and ending at the number in the ()

        label = list(image_dict.keys())[index]
        # In image dict, each class has a spot (1st, 2nd, 3rd...). 'Indexing' lets me pull out class at spot num in []

        encoded_label = np.zeros(class_count)
        # Using zeros method of np package to make array of 0 with length equal to num in ()

        encoded_label[index] = 1.0  # This turns the 0 at spot num in [] to 1.0; this will be different for each class
        encoded_label_dict[label] = encoded_label.reshape(-1, 1)
        # Turing this into matrix of <image class num>x1 and storing as value with key being 'string', look up, in []

    return encoded_label_dict  # 'return' statement again
