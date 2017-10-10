import os
import csv
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as k
from itertools import product
from functools import partial
from keras.layers import Dense
from collections import Counter
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Sequential
from keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# 0 = No data balancing, 1 = Balanced data selction, 2 = Weighted cost function
balance = 0
# 0 = No Fine Tuning, 1 = Data selection adds to training data, 2 = Data selection becomes training set
fineTuning = 0

budget = 10
quality = 0.85


class Model:
    def __init__(self, num_input, num_classes):
        self.num_input = num_input
        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.num_input))
        model.add(Dense(256, kernel_regularizer=l2(0.5)))
        model.add(Dense(256, kernel_regularizer=l2(0.5)))
        model.add(Dense(256, kernel_regularizer=l2(0.5)))
        model.add(Dense(256, kernel_regularizer=l2(0.5)))
        model.add(Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(0.5)))
        return model

    @staticmethod
    def weighted_crossentropy(y_true, y_pred, weights):
        nb_cl = len(weights)
        final_mask = k.zeros_like(y_pred[..., 0])
        y_pred_max = k.max(y_pred, axis=-1)
        y_pred_max = k.expand_dims(y_pred_max, axis=-1)
        y_pred_max_mat = k.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            w = k.cast(weights[c_t, c_p], k.floatx())
            y_p = k.cast(y_pred_max_mat[..., c_p], k.floatx())
            y_t = k.cast(y_pred_max_mat[..., c_t], k.floatx())
            final_mask += w * y_p * y_t
        return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) * final_mask

    def train(self, version, data, labels, batch_size, test_data, test_labels, weights=np.ones((0, 0)), confuse=False):
        if balance == 2:
            weighted_cost = partial(self.weighted_crossentropy, weights=weights)
            self.model.compile(loss=weighted_cost, optimizer=Adam(lr=0.1), metrics=['accuracy'])
        else:
            self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.1), metrics=['accuracy'])
        tb_callback = TensorBoard(log_dir='models/' + str(version) + '/Graph', histogram_freq=0, write_graph=True,
                                 write_images=True)
        self.model.fit(np.asarray(data), np.asarray(labels), batch_size=batch_size, epochs=100, callbacks=[tb_callback],
                       verbose=2)
        if not os.path.isdir('models/' + str(version)):
            os.mkdir('models/' + str(version))
        self.model.save('models/' + str(version) + '/model.h5')
        predictions = self.model.predict(test_data)
        if confuse:
            return self.confusion_matrix(predictions, test_labels)
        else:
            scores = self.model.evaluate(test_data, test_labels, verbose=0)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])
            return scores[1]

    def predict(self, version, data):
        self.model.load_weights('models/' + str(version) + '/model.h5')
        return self.model.predict(data, verbose=0)

    @staticmethod
    def confusion_matrix(predictions, labels):
        y_actu = np.zeros(len(labels))
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] == 1.00:
                    y_actu[i] = j
        y_pred = np.zeros(len(predictions))
        for i in range(len(predictions)):
            y_pred[i] = np.argmax(predictions[i])
        p_labels = pd.Series(y_pred)
        t_labels = pd.Series(y_actu)
        df_confusion = pd.crosstab(t_labels, p_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)
        accuracy = accuracy_score(y_true=y_actu, y_pred=y_pred, normalize=True)
        print('\nAccuracy = ' + str(accuracy) + '\n')
        print(df_confusion)
        print('\n' + str(classification_report(y_actu, y_pred, digits=4)))
        return accuracy


class MNIST:
    def __init__(self, train_file, test_file):
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
        self.predict_x, self.predict_y = [], []

    def reduce_data(self, percentage):
        if balance == 1:
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
                    self.predict_x.append(self.train_x[i])
                    self.predict_y.append(self.train_y[i])
                    delete_indexes.append(i)
            self.train_x = np.delete(self.train_x, delete_indexes, axis=0)
            self.train_y = np.delete(self.train_y, delete_indexes, axis=0)
            self.predict_x, self.predict_y = np.asarray(self.predict_x), np.asarray(self.predict_y)
        else:
            self.train_x, self.predict_x, self.train_y, self.predict_y = train_test_split(self.train_x,
                                                                                          self.train_y,
                                                                                          test_size=percentage)

    def get_weights(self, bootstrap=[], smooth_factor=0):
        if len(bootstrap) == 0:
            labels = bootstrap
        else:
            labels = self.train_y
        temp_y = []
        for i in labels:
            temp_y.append(np.argmax(i))
        counter = Counter(temp_y)
        if smooth_factor > 0:
            p = max(counter.values()) * smooth_factor
            for key in counter.keys():
                counter[key] += p
        majority = max(counter.values())
        weights = {cls: float(majority / count) for cls, count in counter.items()}

        nb_cl = len(weights)
        final_weights = np.ones((nb_cl, nb_cl))
        for class_idx, class_weight in weights.items():
            final_weights[0][class_idx] = class_weight
            final_weights[class_idx][0] = class_weight
        return final_weights

    def check_balance(self):
        temp_y = []
        for i in self.train_y:
            temp_y.append(np.argmax(i))
        counter = Counter(temp_y)
        print('Balance: ' + str(counter))

    def get_bootstraps(self, number_bootstraps, bootstrap_size):
        bootstraps_x, bootstraps_y = [], []
        for _ in range(number_bootstraps):
            bootstrap_x, bootstrap_y = [], []
            if balance == 1:
                pass       # TODO Balanced bootstraps
            else:
                indexs = random.sample(range(len(self.train_x)), bootstrap_size)
                for index in indexs:
                    bootstrap_x.append(self.train_x[index])
                    bootstrap_y.append(self.train_y[index])
            bootstraps_x.append(bootstrap_x)
            bootstraps_y.append(bootstrap_y)
        return bootstraps_x, bootstraps_y

    def increase_data(self, uncertainty, batch):
        # maxes = np.zeros(len(inputs))
        # for i in range(len(inputs)):
        #     maxes[i] = inputs[i][np.argmax(inputs[i])]
        indexes = []
        if fineTuning == 2:
            self.train_x = np.zeros((0, 784))
            self.train_y = np.zeros((0, 10))
        if balance == 1:
            num_to_label = int(batch / 10)
            classification = [[], [], [], [], [], [], [], [], [], []]
            for i in range(len(uncertainty)):
                prediction_class = int(np.argmax(self.predict_y[i]))
                classification[prediction_class].append([uncertainty[i], i])
            for class_maxes in classification:
                c_maxes, big_indexes = [m[0] for m in class_maxes], [n[1] for n in class_maxes]
                for i in range(num_to_label):
                    index = np.where(c_maxes == np.asarray(c_maxes).min())[0][0]
                    class_maxes[index][0] = np.finfo(np.float64).max
                    index = big_indexes[index]
                    self.train_x = np.vstack((self.train_x, [self.predict_x[index]]))
                    self.train_y = np.vstack((self.train_y, [self.predict_y[index]]))
                    indexes.append(index)
            self.predict_x = np.delete(self.predict_x, indexes, axis=0)
            self.predict_y = np.delete(self.predict_y, indexes, axis=0)
        else:
            for i in range(batch):
                index = np.where(uncertainty == uncertainty.min())[0][0]
                uncertainty[index] = np.finfo(np.float64).max
                self.train_x = np.vstack((self.train_x, [self.predict_x[index]]))
                self.train_y = np.vstack((self.train_y, [self.predict_y[index]]))
                indexes.append(index)
            self.predict_x = np.delete(self.predict_x, indexes, axis=0)
            self.predict_y = np.delete(self.predict_y, indexes, axis=0)


def bootstrap_learn(iteration, data, number_bootstraps, bootstrap_size):
    accuracies, uncertainty = [], np.zeros((len(data.predict_x)), dtype=np.float64)
    bootstraps_x, bootstraps_y = data.get_bootstraps(number_bootstraps, bootstrap_size)
    for i in range(len(bootstraps_x)):
        model = Model(784, 10)
        if balance == 2:
            accuracies.append(model.train(str(iteration) + '_' + str(i), bootstraps_x[i], bootstraps_y[i], 10,
                                          data.test_x, data.test_y, data.get_weights(bootstrap=bootstraps_y[i])))
        else:
            accuracies.append(model.train(str(iteration) + '_' + str(i), bootstraps_x[i], bootstraps_y[i], 10,
                                          data.test_x, data.test_y))
        predictions = model.predict(str(iteration) + '_' + str(i), data.predict_x)

        for j in range(len(predictions)):
            uncertainty[j] += np.divide(predictions[j][np.argmax(predictions[j])], number_bootstraps)
            uncertainty[j] = np.multiply(uncertainty[j], (1 - uncertainty[j]))
    accuracy = np.average(np.asarray(accuracies))
    print('Average Accuracy = ' + str(accuracy))
    return accuracy, uncertainty


def main():
    accuracies = []
    questions_asked = 0
    batch = 1

    data = MNIST('mnist_train.csv', 'mnist_test.csv')
    data.reduce_data(0.99)

    accuracy, uncertainty = bootstrap_learn(questions_asked, data, 10, 100)
    questions_asked += 1
    batch *= 2
    accuracies.append(accuracy)

    while accuracy < quality or questions_asked < budget:
        data.increase_data(uncertainty, batch)
        accuracy, uncertainty = bootstrap_learn(questions_asked, data, 10, 100)
        questions_asked += 1
        batch *= 2
        accuracies.append(accuracy)
    print(accuracies)



if __name__ == '__main__':
    main()
