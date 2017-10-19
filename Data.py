import csv
import math
import random
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, balance):
        self.train_x, self.train_y = np.array([]), np.array([])
        self.test_x, self.test_y = np.array([]), np.array([])
        self.predict_x, self.predict_y = np.array([]), np.array([])
        self.balance = balance

    def set_data(self, train_x, train_y):
        self.train_x, self.train_y = train_x, train_y

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
            This method returns a set of weights based on the training labels.

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

    def reduce_data(self, percentage):
        if self.balance:
            predict_x, predict_y = [], []
            elements = math.ceil((len(self.train_y) * percentage / 100) / 10)
            indexes = np.array([])
            for classification in range(10):
                temp_indexs = []
                for i in range(len(self.train_y)):
                    if np.argmax(self.train_y[i]) == classification:
                        temp_indexs.append(i)
                indexes = np.append(indexes, random.sample(temp_indexs, elements))
            indexes = -np.sort(-indexes)
            delete_indexes = []
            for i in range(len(self.train_x) - 1, -1, -1):
                if i not in indexes:
                    predict_x.append(self.train_x[i])
                    predict_y.append(self.train_y[i])
                    delete_indexes.append(i)
            self.train_x = np.delete(self.train_x, delete_indexes, axis=0)
            self.train_y = np.delete(self.train_y, delete_indexes, axis=0)
            self.predict_x, self.predict_y = np.asarray(predict_x), np.asarray(predict_y)
        else:
            self.train_x, self.predict_x, self.train_y, self.predict_y = train_test_split(self.train_x,
                                                                                          self.train_y,
                                                                                          test_size=percentage)

    def get_bootstraps(self, number_bootstraps, bootstrap_size):
        bootstraps = []
        for _ in range(number_bootstraps):
            bootstrap_x, bootstrap_y = [], []
            if self.balance:
                classes = [[],[],[],[],[],[],[],[],[],[]]
                for i in range(len(self.train_x)):
                    classes[np.argmax(self.train_y[i])].append(i)
                for classification in classes:
                    indexs = random.sample(classification, int(bootstrap_size / 10))
                    for index in indexs:
                        bootstrap_x.append(self.train_x[index])
                        bootstrap_y.append(self.train_y[index])
            else:
                indexs = random.sample(range(len(self.train_x)), bootstrap_size)
                for index in indexs:
                    bootstrap_x.append(self.train_x[index])
                    bootstrap_y.append(self.train_y[index])
            data = Data(self.balance)
            data.set_data(np.asarray(bootstrap_x), np.asarray(bootstrap_y))
            bootstraps.append(data)
        return bootstraps

    def increase_data(self, indexes):
        for index in indexes:
            self.train_x = np.vstack((self.train_x, [self.predict_x[index]]))
            self.train_y = np.vstack((self.train_y, [self.predict_y[index]]))
        self.predict_x = np.delete(self.predict_x, indexes, axis=0)
        self.predict_y = np.delete(self.predict_y, indexes, axis=0)
