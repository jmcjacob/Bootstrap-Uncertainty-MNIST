import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import product
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class Model:
    """
        This class handles a single model and contains methods for training and predicting on the model.

        To train a new model use the train method.
            new_model = Model(input_shape, num_classes, save_path)
            model_accuracy = new_model.train(train_x, train_y, test_x, test_y)

        To use an existing model to make predictions use the predict method.
            existing_model = Model(input_shape, num_classes, save_path)
            predictions = existing_model.predict(data)

        To use custom architectures, loss functions or optimisers create a new class that inherits from this class and
        overwrite the "create_model" method with a custom architecture, the "loss" method with a custom loss method and
        the "optimise" method for a custom optimiser.

        The methods "set_loss_params" and "set_optimise_params" can be used to set the parameters that will be used in
        the learning process and can be called to adjust the parameters for training. These methods will need to be
        overwritten for the use of custom loss functions and optimisers.
    """

    def __init__(self, input_shape, num_classes, save_path, verbose=True):
        """
            This initialises creates the class members and creates the neural network model.

        :param input_shape: The shape of the input to the model. (can be a single integer if one dimensional.)
        :param num_classes: The number of classes that the model makes it predictions on.
        :param save_path: The path the model will be saved to. (Can be set to "none" if the model isn't to be saved.)
        :param verbose: Sets the level of verboseness during training.
        """

        # Rests current tensorflow graph
        tf.reset_default_graph()

        # Defines the shape of inputs and classes
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Defines varibles for input, output and the model
        self.X = tf.placeholder('float', [None, self.input_shape])
        self.Y = tf.placeholder('float', [None, self.num_classes])
        self.model = self.create_model()

        self.set_loss_params()
        self.set_optimise_params()

        # Sets the save path for the model and othe parameters
        self.save_path = save_path
        self.verbose = verbose
        self.losses = []

    def __copy__(self):
        """
            A copy constructor that created a new model with the same parameters as the current model.

        :return: A new model with the same parameters as this model.
        """

        model = Model(self.input_shape, self.num_classes, self.save_path, self.verbose)
        model.set_loss_params(weights=self.loss_weights, beta=self.beta)
        model.set_optimise_params(learning_rate=self.learning_rate, decay=self.decay, momentum=self.momentum,
                                  epsilon=self.epsilon, use_locking=self.use_locking, centered=self.centered)
        return model

    def create_model(self):
        """
            Method that initialises a set of weights and biases and returns a neural network.
            This method can be overwritten to add new architectures.

        :return: A Tensorflow graph for a neural network with the given input shape and number of classes.
        """

        # Initialises the weights
        self.weights = {
            'h1': tf.Variable(tf.truncated_normal([self.input_shape, 256])),
            'h2': tf.Variable(tf.truncated_normal([256, 256])),
            'h3': tf.Variable(tf.truncated_normal([256, 256])),
            'out': tf.Variable(tf.truncated_normal([256, self.num_classes]))
        }
        # Initialises the biases
        self.biases = {
            'b1': tf.Variable(tf.truncated_normal([256])),
            'b2': tf.Variable(tf.truncated_normal([256])),
            'b3': tf.Variable(tf.truncated_normal([256])),
            'out': tf.Variable(tf.truncated_normal([self.num_classes]))
        }

        # Initialises the graph acording to an architecture
        layer_1 = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_3 = tf.nn.softsign(tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3']))
        return tf.add(tf.matmul(layer_3, self.weights['out']), self.biases['out'])

    def loss(self):
        """
            This method defines the loss operation that is used to calculate the loss for the model.
            The parameters for this function as set using the "set_loss_params" method.

        :return: A Tensorflow operation to calculate the loss of the model.
        """

        # Checks if weights are present to see if weighted cost function should be used.
        if not self.loss_weights.shape == np.ones((0, 0)).shape:
            nb_cl = len(self.loss_weights)
            final_mask = tf.zeros_like(self.model[..., 0])
            y_pred_max = tf.reduce_max(self.model, axis=-1)
            y_pred_max = tf.expand_dims(y_pred_max, axis=-1)
            y_pred_max_mat = tf.equal(self.model, y_pred_max)

            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                w = tf.cast(self.loss_weights[c_t, c_p], 'float32')
                y_p = tf.cast(y_pred_max_mat[..., c_p], 'float32')
                y_t = tf.cast(y_pred_max_mat[..., c_t], 'float32')
                final_mask += w * y_p * y_t

            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y) * final_mask
        else:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y)

        # Adds l2 regularisation.
        loss += self.beta + tf.nn.l2_loss(self.weights['h2']) + self.beta + tf.nn.l2_loss(self.biases['b2'])
        loss += self.beta + tf.nn.l2_loss(self.weights['h3']) + self.beta + tf.nn.l2_loss(self.biases['b3'])
        loss += self.beta + tf.nn.l2_loss(self.weights['out']) + self.beta + tf.nn.l2_loss(self.biases['out'])
        return tf.reduce_mean(loss)

    def set_loss_params(self, weights=np.ones((0, 0)), beta=0.1):
        """
            This method sets the values of the loss parameters.

        :param weights: The weights to be applied to the loss.
        :param beta: The beta value for applying l2 regularisation. (Can be set to 0 if no regularisation is desired.)
        """

        self.beta = beta
        self.loss_weights = weights

    def optimise(self, loss):
        """
            Defines the optimiser that will be used to train the model.
            The parameters for this function as set using the "set_optimise_params" method.

        :param loss: The loss operation that optimiser will be minimising.
        :return: The operation for model optimisation.
        """

        optimiser = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay,
                                              momentum=self.momentum, use_locking=self.use_locking,
                                              centered=self.centered)
        return optimiser.minimize(loss)

    def set_optimise_params(self, learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False,
                            centered=False):
        """
            This method is used for setting the parameters of the optimiser.

        :param learning_rate: The rate the values are updated.
        :param decay: Discounting factor for the history/coming gradient.
        :param momentum: The momentum scalar.
        :param epsilon: Small value to avoid zero denominator.
        :param use_locking: uses locks for update operation.
        :param centered: Gradients are normalized by the estimated variance of the gradient.
        """

        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_locking = use_locking
        self.centered = centered

    def train(self, train_x, train_y, test_x, test_y, epochs=-1, batch_size=100, val_percent=0.1, confuse=False,
              intervals=10):
        """
            This method will train the system using loss function defined in the "loss" method using the optimiser
            defined in the "optimise" method. The trained model will then be saved in the save path set in the
            initialisation.

        :param train_x: The training data.
        :param train_y: The labels for the training data.
        :param test_x: The testing data.
        :param test_y: The labels for the testing data.
        :param epochs: The number of epochs to train with. (If set to -1 will continue to train until converged.
        :param batch_size: The size of the batches to train the system with.
        :param val_percent: The percentage of training data that should be used as validation data.
        :param confuse: Should a confusion matrix be produced at the end of training.
        :param intervals: The interval for when the system should print information.
        :return: The final accuracy of the model.
        """

        # Splits the training data into training and validation data.
        train_x, val_data, train_y, val_labels = train_test_split(train_x, train_y, test_size=val_percent)

        # Defines the loss, optimiser and prediction operations.
        loss = self.loss()
        optimizer = self.optimise(loss)
        correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initilises the graph varibles and graph saver.
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Begins the Tensorflow session
        with tf.Session() as sess:
            sess.run(init)

            # Splits the training data into batches
            train_x = self.split(train_x, batch_size)
            train_y = self.split(train_y, batch_size)
            num_batches = len(train_x)
            epoch = 0

            # Runs for a number of epochs either from a pre-defined number or runs until converged.
            while not self.converged(epochs) and epoch != epochs:
                avg_loss, avg_acc = 0, 0

                # Runs for each batch.
                for batch in range(num_batches):

                    # Trains on the batch and returns the loss and accuracy for the batch
                    _, cost = sess.run([optimizer, loss],
                                       feed_dict={self.X: np.asarray(train_x[batch]), self.Y: train_y[batch]})
                    avg_loss += cost
                epoch += 1

                # Calculates the validation loss and accuracy.
                val_acc, val_loss = sess.run([accuracy, loss], feed_dict={self.X: val_data, self.Y: val_labels})
                self.losses.append(val_loss)

                # Displays the current parameters for training.
                if self.verbose and epoch % intervals == 0:
                    message = 'Epoch: ' + str(epoch) + ' Loss: ' + '{:.4f}'.format(avg_loss / num_batches)
                    message += ' Validation Accuracy: ' + '{:.3f}'.format(val_acc)
                    message += ' Validation Loss: ' + '{:.4f}'.format(val_loss)
                    print(message)

            # Displays the number of epochs the model was trained in.
            print('Finished at Epoch ' + str(epoch))

            # Computes the Testing accuracy.
            final_acc = sess.run(accuracy, feed_dict={self.X: test_x, self.Y: test_y})
            print('Testing Accuracy: ', str(final_acc))

            # Produces a confusion matrix.
            if confuse:
                predictions = sess.run(tf.nn.softmax(self.model), feed_dict={self.X: test_x})
                self.confusion_matrix(predictions, test_y)

            # Create directory and save the model at save path.
            if self.save_path != 'none':
                if not os.path.isdir(self.save_path):
                    os.makedirs(self.save_path)
                save_path = saver.save(sess, self.save_path + '/model.ckpt')
                print("Model saved in file: %s" % save_path)

            # Returns the accuracy of the model.
            return final_acc

    def predict(self, data, load=False):
        """
            Uses a trained model to make prediction on the inputted data.

        :param data: The input data, must be in an array with each element being a peice of data.
        :return: An array of equal size to data that contain probability predictions for each class.
        """

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            if load:
                saver.restore(sess, self.save_path + '/model.ckpt')
            return sess.run(tf.nn.softmax(self.model), feed_dict={self.X: data})

    @staticmethod
    def confusion_matrix(predictions, labels):
        """
            This method displays the confusion matrix and a classification report.
            The report features metrics precision, recall, F1-score and support.

        :param predictions: An array of the predictions.
        :param labels: An array of one-hot encoded labels.
        """

        # Pre-process the data into arrays of one-dimension labels.
        y_actu = np.zeros(len(labels))
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] == 1.00:
                    y_actu[i] = j
        y_pred = np.zeros(len(predictions))
        for i in range(len(predictions)):
            y_pred[i] = np.argmax(predictions[i])

        # Produce the confusion matrix.
        p_labels = pd.Series(y_pred)
        t_labels = pd.Series(y_actu)
        df_confusion = pd.crosstab(t_labels, p_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)

        # Prints the confusion matrix and classification report.
        print(df_confusion)
        print(classification_report(y_actu, y_pred, digits=4))

    @staticmethod
    def split(data, batch_size):
        """
            This method splits an array of data into batches for training.
            This method will also padd the data so it can be split evenly

        :param data: The array of data that will be split.
        :param batch_size: The size of the batches to split the data.
        :return: An array of batches from the input data.
        """
        # TODO - Multi-dimensional splitting as only one dimensional data is supported.

        # Calculates the number of batches
        num_batches = math.trunc(len(data) / batch_size)

        # Padds the array if the data cant be split equally.
        if len(data) % batch_size != 0:
            data = np.pad(data, [(0, batch_size - (int(len(data) % batch_size))), (0, 0)], 'constant')
            num_batches += 1

        # Retuns the split array.
        return np.split(np.asarray(data), num_batches)

    def converged(self, epochs, min_epochs=50, diff=0.5, converge_len=10):
        """
            Method that checks if the model has reached a point of convergence.

        :param epochs: The number of epochs to check if this funtion should be ignored.
        :param min_epochs: The minimal amount of epochs before convergence checking.
        :param diff: The amout the final values can differ and still count towards convergence.
        :param converge_len: The amount of values to check for convergence.
        :return:
        """

        # Checks the minimum epochs have been met.
        if len(self.losses) > min_epochs and epochs == -1:
            losses = self.losses[-converge_len:]

            # Checks if the last loss values had a close distance to each other.
            for loss in losses[: (converge_len - 1)]:
                if abs(losses[-1] - loss) > diff:
                    return False
            return True
        return False
