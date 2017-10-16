import csv
import numpy as np
from collections import Counter


class Data:
    def __init__(self):
        self.train_x, self.train_y = np.array([]), np.array([])
        self.test_x, self.test_y = np.array([]), np.array([])

    def load_data(self, train_file, test_file):
        """
            The method used to load the training and testing data and labels from an inputted location as members of
            the class.

        :param train_file: The data source of the training data.
        :param test_file: The data source of the testing data.
        """
        train_x, train_y = [], []
        test_x, test_y = [], []
        with open(train_file) as file:
            reader = csv.reader(file)
            for row in reader:
                label = np.zeros(10)
                label[int(row[0])] = 1
                train_y.append(label)
                train_x.append(list(map(int, row[1:])))
        print('Training Data Loaded from file')

        with open(test_file) as file:
            reader = csv.reader(file)
            for row in reader:
                label = np.zeros(10)
                label[int(row[0])] = 1
                test_y.append(label)
                test_x.append(list(map(int, row[1:])))
        print('Testing Data Loaded from file')

        self.train_x, self.train_y = np.asarray(train_x), np.asarray(train_y)
        self.test_x, self.test_y = np.asarray(test_x), np.asarray(test_y)

    def get_weights(self):
        """
            This method returns a set of weights based on the training labels

        :return: A set of weights.
        """

        temp_y = []
        for i in self.train_y:
            temp_y.append(np.argmax(i))
        counter = Counter(temp_y)
        majority = max(counter.values())
        weights = {cls: float(majority / count) for cls, count in counter.items()}

        nb_cl = len(weights)
        final_weights = np.ones((nb_cl, nb_cl))
        for class_idx, class_weight in weights.items():
            final_weights[0][class_idx] = class_weight
            final_weights[class_idx][0] = class_weight
        return final_weights
