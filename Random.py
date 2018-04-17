import copy
import random
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

    def run(self, batch_size):
        accuracies = []

        self.accuracy, _ = self.train_predict(self.data)
        accuracies.append(self.accuracy)

        while self.budget != self.questions_asked and self.quality > self.accuracy:
            indices = list(range(len(self.data.data_y)))
            self.data.increase_data(random.sample(indices, batch_size))
            self.accuracy, _ = self.train_predict(self.data)
            accuracies.append(self.accuracy)
            self.questions_asked = self.questions_asked + 1
        return accuracies

