import numpy as np
from network_tuning import *
from analyze_data import *


# Below is the Neural Network class with its functions for neural network creation, training, and testing.

class NeuralNetwork:
    """
    This class creates instances of a neural network for classifying images. The layer size,
    the number of hidden layers, and the classes used for image classification are
    customizable.

    Upon instance creation, the class initiates layer sizes, weights, and biases.
    This class has a 'train' method for teaching network instances which kinds of images belong to
    which image classes. It has a 'test' method for testing the accuracy of a trained neural network
    instance in classifying new images (of the same kind used in training) with the learned image
    classes.

    This class has a 'feedforward' method for converting images to image class probabilities used to make
    classification predictions, and it has a 'feedback' method used to update layer weights and biases
    during training based on the calculated loss between the prediction and the true class label
    of the image.

    The activity of neural network instances of this class is modulated by 2 mathematical formulations
    of 2 signaling molecules: ccl5 and wnt. In biological neural networks, these govern the release of
    the excitatory neurotransmitter glutamate and the stability of synapses between neurons, respectively.
    These signaling molecules are passed into neural network instances as 'deltas', differences between
    the normal levels and those found in the brain of ASD patients. Both 'ccl5_delta' and 'wnt_delta'
    default to 0 to simulate normal brain activity.
    """

    def __init__(self, class_list, image_vector_size, hidden_layer_number, ccl5_delta=0, wnt_delta=0):
        """
        This function initializes neural network layer sizes, weights, and biases.

        It takes a class list, used to set the size of the output layer for comparison to the encoded
        labels of images, which also equal class list length. It takes an image vector size, which serves
        as the input layer size and is used to set the sizes of the hidden layers. It takes a hidden layer
        number, used to create a matching number of hidden layers in the neural network, equal in size
        to the input layer (i.e., image vector size).

        All hidden layers start equal in size to the input layer to simulate the sparse encoding of visual signals
        seen in the visual pathway of humans and other primates. That is to say, each neuron (node) in a
        visual brain region or regional subdivision (layer) holds a unique section (pixel) of the visual
        scene (image), represented as an activity/firing pattern (number). As information is passed from
        one layer to the next, each neuron (node) conveys its activity (number) to a downstream neuron
        one neuron that will primarily represent its unique section (pixel) of the visual scene and to one
        neuron representing every other unique section in the visual scene (image). This is an idealized
        unique 1-to-unique 1 relationship preventing the transfer of redundant information, which is the
        posited goal of spare encoding in biological neural networks.

        Wnt supports synaptic stability and higher/lower levels have been shown to lead to a higher/lower
        number of synapses formed between a given neuron and its downstream partners. In this function,
        'wnt_delta' stores a fold increase (e.g, 1.4 found in the brain of ASD patients) or decrease
        compared to normal wnt levels. This value is used as a scaling factor for layer size: it is
        multiplied by the size of the upstream layer to get the number of additional nodes that will be
        added or subtracted from the size of the upstream layer to get the size of the downstream layer.
        Scaling up creates redundancy, where some neurons in the upstream layer convey their activity to
        more than one neuron that will, in turn, primarily represent their unique section of the visual
        scene (i.e., multiple nodes representing the same pixel). Whereas, scaling down makes encoding
        even sparser: several neurons in the upstream layer converge on a single neuron in the downstream
        layer (i.e., representations of multiple pixels are pooled into the same node).

        Here, the 'ccl5_delta' value is stored for later use in get_release_maps(), see network_tuning.py.

        Parameters
        ----------
        class_list
        image_vector_size
        hidden_layer_number
        ccl5_delta
        wnt_delta
        """

        # Initializing layer sizes
        self.layer_sizes = [image_vector_size]
        for layer in range(hidden_layer_number):
            additional_nodes = int(self.layer_sizes[-1] * wnt_delta)
            layer_size = self.layer_sizes[-1] + additional_nodes
            self.layer_sizes.append(layer_size)
        self.layer_sizes.append(len(class_list))

        # Initializing layer weights, biases
        self.layer_weights = []
        self.layer_biases = []
        for layer in range(hidden_layer_number + 1):
            layer_weights = np.random.randn(self.layer_sizes[layer + 1], self.layer_sizes[layer])
            self.layer_weights.append(layer_weights)
            layer_biases = np.random.randn(self.layer_sizes[layer + 1], 1)
            self.layer_biases.append(layer_biases)

        # Initializing layer modulation
        self.ccl5_delta = ccl5_delta

    def feedforward(self, image_vector, use_release_maps):
        """
        This function passes the pixel values of the image vector (input layer) through
        the hidden layers to the output layer, with the inputs from one layer to the next
        scaled by distinct weight matrices, and return a list of layer activities.

        A sigmoid function is applied to the activity calculated for each hidden layer, and
        a softmax functon is applied to the activity calculated for the output layer. These
        are important for comparing the output of the network to the encoded label of images,
        see network_tuning.py.

        The use of release maps simulating stochastic neurotransmitter release for dynamic
        weighting vs fixed weighting of synaptic inputs (see network_tuning.py) can be
        turned on/off by setting 'use_release_maps' to True/False.

        Parameters
        ----------
        image_vector
        use_release_maps

        Returns
        -------
        layer_activities
        """

        # Creating maps of release probabilities
        release_maps = get_release_maps(self.layer_weights, self.ccl5_delta)

        # Calculating layer activities
        layer_activities = [image_vector]
        for layer in range(len(self.layer_weights)):
            if use_release_maps:
                weights = self.layer_weights[layer] * release_maps[layer]
            else:
                weights = self.layer_weights[layer]
            layer_activity = np.dot(weights, layer_activities[-1]) + self.layer_biases[layer]
            if layer == len(self.layer_weights):
                normalized_activity = apply_softmax(layer_activity)
            else:
                normalized_activity = apply_sigmoid(layer_activity)
            layer_activities.append(normalized_activity)

        return layer_activities

    def feedback(self, encoded_label, layer_activities, learn_r):
        """
        This function updates the weights and biases for connections between layers of the
        network based on the differences between image class probabilities held by nodes in
        the output layer and the encoded image label.

        It takes an encoded image label and subtracts it from the output layer activity produced
        by the paired image, grabbed from a list of layer activities also passed in, to get a delta
        (difference) matrix. The values of this matrix are propagated backwards by multiplying it
        by the transposition of the weight matrix the output layer and the last hidden layer. These
        values are then multiplied by the result of [the activity of the last hidden layer * (1 - the
        activity of the last hidden layer)] to get a delta matrix for the last hidden layer (reflecting
        the contribution each of its node make to the differences in the output layer vs the encoded
        label). These delta values are back-propagated to each hidden layer using the same strategy.

        The deltas for each layer are clipped by first making them into one long vector (1D matrix),
        adding the square of each value up, and taking the square root of the sum for the Euclidean
        norm (single value). If the Euclidean norm is greater than a pre-set max gradient norm value,
        a clipping factor is calculated by dividing the max gradient norm by the Euclidean norm. The
        deltas for each layer are then multiplied by this clipping factor, making them smaller and
        easier to handle for backpropagation.

        The deltas for each layer used to update their associated weights by multiplying the layer delta
        by the transposition of the layer activity. The values in the resulting matrix are scaled by
        multiplying it by a pre-set learning rate ('learn_r') passed in. This scaled matrix of weight
        updates is then subtracted from the existing layer weight matrix.

        Layer deltas are also used to update the biases associated with each layer by first multiplying
        the layer delta values by the learning rate (to scale them), then subtracting the existing layer
        biases matrix by the scaled layer delta matrix.

        Cross-entropy loss (difference) of neural network output layer activity vs encoded image label
        is calculated by taking the natural log of node values (the smaller the numerical input, the
        larger the negative numerical output), which is the inverse of Euler's number**node value,
        applied to layer activities via the Sigmoid function during feedforward(). The loss matrix,
        equal in size to the output layer, is multiplied by the encoded image label to zero out
        values at all nodes representing incorrect image classes and preserving the value at the node
        representing the correct/true image class. This matrix is summed and converted to a positive
        value before being returned.

        Parameters
        ----------
        encoded_label
        layer_activities
        learn_r

        Returns
        -------
        loss
        """

        # Calculating differences in output matrix vs label matrix
        output_delta = layer_activities[-1] - encoded_label

        # Calculating loss based on output vs label
        loss = -np.sum(encoded_label * np.log(layer_activities[-1] + 1e-10))

        # Backpropagation of differences in output matrix vs label matrix
        layer_deltas = [output_delta]
        for layer in range(len(self.layer_weights) - 1):
            back_layer = -1 * (layer + 1)
            delta = np.dot(self.layer_weights[back_layer].T, layer_deltas[-1])
            minuend = np.ones(self.layer_sizes[back_layer - 1]).reshape(-1, 1)
            layer_delta = delta * (layer_activities[back_layer - 1] * (minuend - layer_activities[back_layer - 1]))
            layer_deltas.append(layer_delta)

        # Clipping gradients to reduce exploding or vanishing weights
        max_gradient_norm = 1.0
        all_gradients = np.concatenate([delta.flatten() for delta in layer_deltas])
        gradient_norm = np.linalg.norm(all_gradients)

        if gradient_norm > max_gradient_norm:
            clipping_factor = max_gradient_norm / gradient_norm
            for index in range(len(layer_deltas)):
                layer_deltas[index] *= clipping_factor

        # Updating layer weights, biases
        for layer in range(len(self.layer_weights)):
            back_layer = -1 * (layer + 1)
            self.layer_weights[back_layer] -= learn_r * np.dot(layer_deltas[layer], layer_activities[back_layer - 1].T)
            self.layer_biases[back_layer] -= learn_r * layer_deltas[layer]

        return loss

    def train(self, training_image_dict, encoded_label_dict, learn_r, sessions):
        """
        This function trains the neural network to classify images according to the specific class
        labels with which the images are paired.

        Predictions of image class are made using feedforward(), and weights applied
        to the values passed between layers to make predictions are updated using feedback().

        This function takes a session count and iterates over it for a specified number of training sessions.
        It also takes a dictionary of image class name (key)-list of image vectors (value) pairs curated
        for training and for each session, each image class key and its paired list of image vectors
        is pulled out. The image vectors are individually passed into feedforward(), see above,
        and a list of layer activities is returned.

        This function takes an encoded label dictionary with image class name(key)-encoded class label
        pairs. The current image class is given as the key to this dictionary to get the encoded
        label for the image class. The list of layer activities and the encoded class label for
        the current image class is passed into feedback(), see above, along with a pre-set
        learning rate given as an argument to this train() function, and returns loss. The loss
        returned for each image is added to a total loss, with which the average loss for each
        session is calculated and printed, along with the session number.

        Loss values for each session are also stored in a list with its corresponding session number
        stored in a sessions number list. The loss over time list is used in conjunction with the
        list of session numbers to plot mean loss over sessions with plot_loss_over_time(),see
        analyze_data.py. This plot is rendered after training is complete.

        Parameters
        ----------
        training_image_dict
        encoded_label_dict
        learn_r
        sessions
        """

        loss_over_time = []
        session_list = []
        for session in range(sessions):
            total_loss = 0

            for image_class in training_image_dict.keys():
                image_list = training_image_dict[image_class]
                encoded_label = encoded_label_dict[image_class]

                for image_vector in image_list:
                    layer_activities = self.feedforward(image_vector, use_release_maps=True)
                    loss = self.feedback(encoded_label, layer_activities, learn_r)
                    total_loss += loss

                    loss_over_time.append(loss)
                    session_list.append(session + 1)

            avg_loss = total_loss / len(training_image_dict.values())
            print(f"Session: {session + 1} of {sessions}, Loss: {avg_loss:.4f}")

        print("Training complete")
        plot_loss_over_time(loss_over_time, session_list)

    def test(self, test_image_dict, encoded_label_dict, use_release_maps):
        """
        This function tests the neural network on classifying new image (distinct for those in the
        training image set) according to the image classes the network learned in training.

        It takes a dictionary of image class name (key)-list of image vectors (value) pairs curated
        for testing. Each image class key and its paired list of image vectors is pulled out and the
        image vectors are individually passed into feedforward(), see above, and a list of layer
        activities is returned. The output layer matrix is converted to a matrix of equal size in which
        every cell contains the index (number) of the node with the highest activity value/image class
        probability.

        This function takes an encoded label dictionary with image class name(key)-encoded class label
        pairs. The current image class is given as the key to this dictionary to get the encoded
        label for the image class. The encoded image label is also converted to a matrix of equal size
        in which every cell contains the index of the node with 1 (all other nodes/cells of the encoded
        label are 0).

        The converted output layer activity and the converted encoded image label are directly compared.
        If they match, the trial is considered correct and a 1 is added to the total correct trial count.
        After all images have been tested, the total correct trial count is divided by the total number
        of images tested (times 100) to get a prediction accuracy percentage, which is then printed.

        Parameters
        ----------
        test_image_dict
        encoded_label_dict
        use_release_maps
        """

        num_correct = 0

        for image_class in test_image_dict.keys():
            image_list = test_image_dict[image_class]
            encoded_label = encoded_label_dict[image_class]
            label = np.argmax(encoded_label)

            for image_vector in image_list:
                layer_activities = self.feedforward(image_vector, use_release_maps)
                prediction = np.argmax(layer_activities[0])

                if prediction.all() == label.all():
                    num_correct += 1

        all_images = [len(x) for x in test_image_dict.values()]
        accuracy = (num_correct / sum(all_images)) * 100

        print(f"Test Accuracy: {accuracy:.2f}%")
