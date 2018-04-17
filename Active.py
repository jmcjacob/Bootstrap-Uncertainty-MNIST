import copy
import numpy as np


class Active:
    """
        The class used the Data class and the Model class to simulate Active learning using a labelled dataset.

        To run the object needs to be created by using the constructor and then using the run method:
            active = Active(data_object, model_object, budget, target_accuracy)
            active.run(num_bootstraps, size_bootstraps, batch_size)

        This class uses the bootstrap uncertainty algorithm from Mozafari et al (2015) as described in the paper:
        https://web.eecs.umich.edu/~mozafari/papers/vldb_2015_crowd.pdf
    """

    def __init__(self, data, model, budget, quality):
        """
            The constructor sets the class members values according to what has been inputted.

        :param data: The data object that contains training, testing and prediction data.
        :param model: A base model that new models will use to copy parameters from.
        :param budget: The amount of questions that can be asked by the system.
        :param quality: The target accuracy of the system.
        """

        self.data = data
        self.model = model
        self.budget = budget
        self.quality = quality
        self.accuracy = 0.0
        self.questions_asked = 0
        self.values = np.zeros((len(self.data.predict_x)))

    def new_model(self):
        """
            This method creates a new model using the copy constructor of the model in the object.
        :return: A model with the same parameters as the class member model.
        """

        return copy.copy(self.model)

    def train_predict(self, data, verbose=True):
        """
            This method created a new model and trains it with the inputted data object, predictions are then made by
            the trained model on the prediction data set and are returned along with the accuracy.

        :param data: The data object that will be used to train the new model.
        :param verbose: If additional print statments should be called.
        :return: The accuracy of the model and a list of predictions made by the trained model on the prediction data.
        """

        # Additional print information
        if verbose:
            print('\nQuestions asked: ' + str(self.questions_asked))
            print('Data length: ' + str(len(data.train_x)))

        # Creates a copy of the class member model.
        model = self.new_model()

        # Sets the model to use weights if class member model has custom weights.
        if not self.model.loss_weights.shape == np.ones((0, 0)).shape:
            model.set_loss_params(weights=data.get_weights())

        # Trains the model and computes accuracy and returns the accuracy and the predictions of the model.
        accuracy = model.train(data.train_x, data.train_y, self.data.test_x, self.data.test_y)
        return accuracy, model.predict(self.data.predict_x)

    def ranking(self, predictions, number_bootstraps):
        """
            This method produces the ranking that determine which data will be taken form the prediction set to the
            training set.
            This method can be overridden to implement a custom ranking method.

        :param predictions: The predictions from a trained model on the prediction data set.
        :param number_bootstraps: The number of bootstraps the predictions have been taken from.
        """

        for j in range(len(predictions)):
            index = np.argmax(predictions[j])
            temp = predictions[j][index]
            self.values[j] += np.divide(temp, number_bootstraps)
            self.values[j] = np.multiply(self.values[j], (1 - self.values[j]))

    def get_indexes(self, batch_size):
        """
            This method returns a list of indices of prediction data that should be moved to the training data.

        :param batch_size: The amount of data to be selected.
        :return: A list of indexes.
        """

        indexes = []
        if self.data.balance:

            # Selects the minimum values from the rankings picking an equal number per classification.
            num_to_label = int(batch_size / 10)
            classification = [[], [], [], [], [], [], [], [], [], []]
            for i in range(len(self.values)):
                prediction_class = int(np.argmax(self.data.predict_y[i]))
                classification[prediction_class].append([self.values[i], i])
            for class_maxes in classification:
                c_maxes, big_indexes = [m[0] for m in class_maxes], [n[1] for n in class_maxes]
                for i in range(num_to_label):
                    index = np.where(c_maxes == np.asarray(c_maxes).min())[0][0]
                    class_maxes[index][0] = np.finfo(np.float64).max
                    index = big_indexes[index]
                    indexes.append(index)
        else:

            # Selects the minimum values from the rankings.
            for i in range(batch_size):
                index = np.where(self.values == self.values.min())[0][0]
                self.values[index] = np.finfo(np.float64).max
                indexes.append(index)
        return indexes

    def run(self, number_bootstraps, bootstrap_size, batch_size):
        """
            This method extracts a number of bootstraps from the training data and trains a model from each one. The
            trained models then make predictions on the prediction data and moved it to the training data. This is
            repeated until a certain number of questions have been asked or a target accuracy has been achieved.

        :param number_bootstraps: The number of bootstraps taken from the training data.
        :param bootstrap_size: The amount of data in each bootstrap.
        :param batch_size: The amount of unlabelled data to add to the training data per question.
        :return: The list of accuracies of the model each time the data was increased.
        """

        accuracies = []

        # Initially trains a model
        self.accuracy, _ = self.train_predict(self.data)
        accuracies.append(self.accuracy)

        # Runs into conditions have been met.
        while self.budget != self.questions_asked and self.quality > self.accuracy:
            self.values = np.zeros((len(self.data.predict_x)))

            # Extracts bootstraps and runs for each bootstrap.
            bootstraps = self.data.get_bootstraps(number_bootstraps, bootstrap_size)
            for i in range(len(bootstraps)):
                print('\nBootstrap: ' + str(i))

                # Trains a model using a single bootstrap and makes predictions on the prediction data.
                _, predictions = self.train_predict(bootstraps[i], verbose=False)

                # Add this models predictions to the rankings.
                self.ranking(predictions, number_bootstraps)

            # Extracts the indexes from the rankings and increases the training data with the indexes.
            indices = list(range(len(self.data.data_y)))
            indexes = self.get_indexes(batch_size)
            self.data.increase_data(indexes)

            # Increases the number of questions and determines an accuracy with the new training data.
            self.questions_asked = self.questions_asked + 1
            self.accuracy, _ = self.train_predict(self.data)
            accuracies.append(self.accuracy)

        return accuracies
