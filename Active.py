import copy
import numpy as np
import tensorflow as tf


class Active:
    def __init__(self, data, model, budget, quality):
        self.data = data
        self.model = model
        self.budget = budget
        self.quality = quality
        self.accuracy = 0.0
        self.questions_asked = 0

    def new_model(self):
        return copy.copy(self.model)

    def train_predict(self, data, verbose=True):
        if verbose:
            print('\nQuestions asked: ' + str(self.questions_asked))
            print('Data length: ' + str(len(data.train_x)))
        model = self.new_model()
        if not self.model.loss_weights.shape == np.ones((0, 0)).shape:
            model.set_loss_params(weights=data.get_weights())
        accuracy = model.train(data.train_x, data.train_y, self.data.test_x, self.data.test_y)
        return accuracy, model.predict(self.data.predict_x)

    def ranking(self, values, predictions, number_bootstraps):
        for j in range(len(predictions)):
            index = np.argmax(predictions[j])
            temp = predictions[j][index]
            values[j] += np.divide(temp, number_bootstraps)
            values[j] = np.multiply(values[j], (1 - values[j]))
        return values

    def get_indexes(self, values, batch_size):
        indexes = []
        if self.data.balance:
            num_to_label = int(batch_size / 10)
            classification = [[], [], [], [], [], [], [], [], [], []]
            for i in range(len(values)):
                prediction_class = int(np.argmax(self.predict_y[i]))
                classification[prediction_class].append([values[i], i])
            for class_maxes in classification:
                c_maxes, big_indexes = [m[0] for m in class_maxes], [n[1] for n in class_maxes]
                for i in range(num_to_label):
                    index = np.where(c_maxes == np.asarray(c_maxes).min())[0][0]
                    class_maxes[index][0] = np.finfo(np.float64).max
                    index = big_indexes[index]
                    indexes.append(index)
        else:
            for i in range(batch_size):
                index = np.where(values == values.min())[0][0]
                values[index] = np.finfo(np.float64).max
                indexes.append(index)
        return indexes

    def run(self, number_bootstraps, bootstrap_size, batch_size):
        accuracies = []
        while self.budget != self.questions_asked and self.quality > self.accuracy:
            self.accuracy, _ = self.train_predict(self.data)

            values = np.zeros((len(self.data.predict_x)))

            bootstraps = self.data.get_bootstraps(number_bootstraps, bootstrap_size)
            for i in range(len(bootstraps)):
                print('\nBootstrap: ' + str(i))
                _, predictions = self.train_predict(bootstraps[i], verbose=False)
                values = self.ranking(values, predictions, number_bootstraps)

            indexes = self.get_indexes(values, batch_size)
            self.data.increase_data(indexes)

            self.accuracy, _ = self.train_predict(self.data)
            accuracies.append(self.accuracy)
            self.questions_asked += 1

        return accuracies
