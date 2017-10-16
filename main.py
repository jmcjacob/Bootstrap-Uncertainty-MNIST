import csv
import sys
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as k
from itertools import product
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# 0 = No data balancing, 1 = Balanced data selction, 2 = Weighted cost function
balance = int(sys.argv[1])
budget = 10
quality = 0.85


class Model:
    def __init__(self, num_input, num_classes):
        tf.reset_default_graph()
        self.num_input = num_input
        self.num_classes = num_classes
        self.X = tf.placeholder('float', [None, self.num_input])
        self.Y = tf.placeholder('float', [None, self.num_classes])
        self.weights, self.biases = {}, {}
        self.model = self.create_model()
        self.losses = []

    def create_model(self):
        self.weights = {
            'h1': tf.Variable(tf.truncated_normal([self.num_input, 256])),
            'h2': tf.Variable(tf.truncated_normal([256, 256])),
            'h3': tf.Variable(tf.truncated_normal([256, 256])),
            'out': tf.Variable(tf.truncated_normal([256, self.num_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.truncated_normal([256])),
            'b2': tf.Variable(tf.truncated_normal([256])),
            'b3': tf.Variable(tf.truncated_normal([256])),
            'out': tf.Variable(tf.truncated_normal([self.num_classes]))
        }
        layer_1 = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_3 = tf.nn.softsign(tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3']))
        return tf.add(tf.matmul(layer_3, self.weights['out']), self.biases['out'])

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

    def train(self, version, data, labels, batch_size, test_data, test_labels, predict_data, weights=np.ones((0, 0)),
              confuse=False):
        data, val_data, labels, val_labels = train_test_split(data, labels, test_size=0.1)

        print(version)
        beta = 0.1
        if balance == 2:
            loss = self.weighted_crossentropy(y_true=self.Y, y_pred=self.model, weights=weights)
        else:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y)
        loss += beta + tf.nn.l2_loss(self.weights['h2']) + beta + tf.nn.l2_loss(self.biases['b2'])
        loss += beta + tf.nn.l2_loss(self.weights['h3']) + beta + tf.nn.l2_loss(self.biases['b3'])
        loss += beta + tf.nn.l2_loss(self.weights['out']) + beta + tf.nn.l2_loss(self.biases['out'])
        loss = tf.reduce_mean(loss)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)
        correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()
        # saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            data = self.split(data, batch_size)
            labels = self.split(labels, batch_size)
            num_batches = len(data)
            epoch = 0
            while not self.converged():
                avg_loss, avg_acc = 0, 0
                for batch in range(num_batches):
                    _, cost, acc = sess.run([optimizer, loss, accuracy],
                                            feed_dict={self.X: np.asarray(data[batch]), self.Y: labels[batch]})
                    avg_loss += cost
                    avg_acc += acc
                epoch += 1
                val_acc, val_loss = sess.run([accuracy, loss], feed_dict={self.X: val_data, self.Y: val_labels})
                self.losses.append(val_loss)
                # if epoch % 10 == 0:
                #     message = 'Epoch: ' + str(epoch) + ' Validation Accuracy: ' + '{:.3f}'.format(val_acc)
                #     message += ' Validation Loss: ' + '{:.4f}'.format(val_loss)
                #     print(message)
            print('Finished at Epoch ' + str(epoch))
            final_acc = sess.run(accuracy, feed_dict={self.X: test_data, self.Y: test_labels})
            predictions = sess.run(tf.nn.softmax(self.model), feed_dict={self.X: test_data})
            if confuse:
                self.confusion_matrix(predictions, test_labels)
            else:
                print('Testing Accuracy: ', str(final_acc))
            # if not os.path.isdir('model/' + str(version)):
            #     os.mkdir('model/' + str(version))
            # save_path = saver.save(sess, 'model/' + str(version) + '/model.ckpt')
            # print("Model saved in file: %s" % save_path)
            if predict_data == 'skip':
                return final_acc
            else:
                return sess.run(tf.nn.softmax(self.model), feed_dict={self.X: predict_data})

    def predict_batch(self, version, batch_data):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, 'model/' + str(version) + '/model.ckpt')
            # for data in batch_data:
            #     predictions.append(sess.run(self.model, feed_dict={self.X: [batch_data]})[0])
            return sess.run(tf.nn.softmax(self.model), feed_dict={self.X: batch_data})

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
        print('Accuracy = ' + str(accuracy) + '\n')
        print(df_confusion)
        print(classification_report(y_actu, y_pred, digits=4))
        return accuracy

    @staticmethod
    def split(input_array, batch_size):
        nu_batches = math.trunc(len(input_array) / batch_size)
        if len(input_array) % batch_size != 0:
            data = np.pad(input_array, [(0, batch_size - (int(len(input_array) % batch_size))), (0, 0)], 'constant')
            return np.split(data, nu_batches + 1)
        else:
            return np.split(np.asarray(input_array), nu_batches)

    def converged(self, min_epochs=50, diff=1.0, converge_len=5):
        if len(self.losses) > min_epochs:
            losses = self.losses[-converge_len :]
            for loss in losses[: (converge_len - 1)]:
                if abs(losses[-1] - loss) > diff:
                    return False
            return True
        return False


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
        if len(bootstrap) != 0:
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

    def check_balance(self, input=[]):
        temp_y = []
        if len(input) == 0:
            data = self.train_y
        else:
            data = input
        for i in data:
            temp_y.append(np.argmax(i))
        counter = Counter(temp_y)
        print('Balance: ' + str(counter))

    def get_bootstraps(self, number_bootstraps, bootstrap_size):
        bootstraps_x, bootstraps_y = [], []
        for _ in range(number_bootstraps):
            bootstrap_x, bootstrap_y = [], []
            if balance == 1:
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
            bootstraps_x.append(bootstrap_x)
            bootstraps_y.append(bootstrap_y)
        return bootstraps_x, bootstraps_y

    def increase_data(self, uncertainty, batch):
        # maxes = np.zeros(len(inputs))
        # for i in range(len(inputs)):
        #     maxes[i] = inputs[i][np.argmax(inputs[i])]
        indexes = []
        if balance == 1 and batch % 10 == 0:  # TODO: look to how this can be implemented fairly.
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


def bootstrap_learn(iteration, data, number_bootstraps, bootstrap_size, batch):
    uncertainty = np.zeros((len(data.predict_x)), dtype=np.float64)
    bootstraps_x, bootstraps_y = data.get_bootstraps(number_bootstraps, bootstrap_size)
    for i in range(len(bootstraps_x)):
        model = Model(784, 10)
        if balance == 2:
            predictions = model.train(str(iteration) + '_' + str(i), bootstraps_x[i], bootstraps_y[i], 100,
                                      data.test_x, data.test_y, data.predict_x,
                                      data.get_weights(bootstrap=bootstraps_y[i]))
        else:
            predictions = model.train(str(iteration) + '_' + str(i), bootstraps_x[i], bootstraps_y[i], 100,
                                      data.test_x, data.test_y, data.predict_x)

        for j in range(len(predictions)):
            index = np.argmax(predictions[j])
            temp = predictions[j][index]
            uncertainty[j] += np.divide(temp, number_bootstraps)
            uncertainty[j] = np.multiply(uncertainty[j], (1 - uncertainty[j]))
    print('\n---------------------------------------------------------------------------------------------------------')
    data.increase_data(uncertainty, batch)
    print('Size of dataset = ' + str(len(data.train_x)))
    if balance == 2:
        accuracy = Model(784, 10).train(iteration, data.train_x, data.train_y, 100, data.test_x, data.test_y, 'skip',
                                        data.get_weights(), confuse=True)
    else:
        accuracy = Model(784, 10).train(iteration, data.train_x, data.train_y, 100, data.test_x, data.test_y, 'skip',
                                        confuse=True)
    return accuracy


def main():
    accuracies = []
    accuracy = 0
    questions_asked = 0
    batch = 100

    data = MNIST('mnist_train.csv', 'mnist_test.csv')
    if balance == 2:
        print('Original Accuracy: ' + str(Model(784, 10).train('Original', data.train_x, data.train_y, 10, data.test_x,
                                                               data.test_y, 'skip', data.get_weights())))
    else:
        print('Original Accuracy: ' + str(Model(784, 10).train('Original', data.train_x, data.train_y, 10, data.test_x,
                                                               data.test_y, 'skip')))

    data.reduce_data(0.99)

    while accuracy < quality and questions_asked != budget:
        accuracy = bootstrap_learn(questions_asked, data, 50, 500, batch)
        questions_asked += 1
        accuracies.append(accuracy)
    print(accuracies)


if __name__ == '__main__':
    main()
