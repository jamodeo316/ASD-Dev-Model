import numpy as np


# Below are non-class-specific functions used for neural network training, testing, and bio simulation.

def get_release_maps(layer_weights, ccl5_delta):
    """
    This function takes the dimensions of the weight matrix for each layer
    and creates a matrix of random values with the same dimensions. The random
    values are selected from a poisson distribution centered on the value given
    as the ccl5 delta. A list of so-called release maps is returned.

    Poisson distributions have been shown to simulate the stochastic release of
    neurotransmitters by synapses. These release matrices/maps can be multiplied
    by the weight matrices for connections between layers to scale the numbers
    passed between nodes in those layers to biological magnitudes.

    Parameters
    ----------
    layer_weights
    ccl5_delta

    Returns
    -------
    release_maps
    """

    synaptic_potential = 100 * (1 + ccl5_delta)

    release_maps = []
    for weights in layer_weights:
        release_map = np.random.poisson(synaptic_potential, size=(weights.shape[0], weights.shape[1])) / 100
        release_maps.append(release_map)
    return release_maps


def apply_sigmoid(layer):
    """
    This function takes a layer of the network and applies a sigmoid funtion to the
    values in each of its nodes. The sigmoid function puts the node value as the negative
    exponent of Euler's number (1 / 2.718**node value), adds 1 to this, then divides the
    result (which is between 1-2.718) out of 1. This forces each node value between 0-1
    while maintaining their relative differences. The updated layer is returned.

    This is important for comparing the output of the network to the encoded label of images,
    in which the image class that the image belongs to is represented as a sequence of 0s and 1s,
    after apply a softmax function (discussed below) to the output layer.

    Parameters
    ----------
    layer

    Returns
    -------
    constrained_activity
    """

    constrained_activity = 1 / (1 + np.exp(-layer))
    return constrained_activity


def apply_softmax(layer):
    """
    This function takes the output layer of the network, finds the value of the largest
    node, subtracts each node value by this max value, and places the negative result as the
    exponent of Euler's number (1 / 2.718**node value). These values are then each divided
    by the sum of all the values to determine the percentage to which they each contribute
    to the sum. The updated layer is returned.

    If each node in the output layer represents an image class, a 1 at a specific node and 0s
    at the others can be thought of a 100%-confident assignment of the image class, represented
    by the position of that specific node, to the image fed into the network. The actual values
    at output layer node are predominately between 0-1, with the largest valued node being the
    image class the network thinks the image most likely belongs to (i.e., image class
    probabilities). These probabilities are compared to the encoded label. In training,
    the difference is measured and layer weights are updated accordingly. In testing,
    the highest probability is taken as the network's prediction of the image class to
    which a given image belongs.

    Parameters
    ----------
    layer

    Returns
    -------
    image_class_probs
    """

    node_deltas = np.exp(layer - np.max(layer))
    image_class_probs = node_deltas / node_deltas.sum(axis=0)
    return image_class_probs
