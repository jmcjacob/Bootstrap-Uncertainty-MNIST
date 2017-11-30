import os
from Data import Data
from Model import Model
from Active import Active


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    accuracies1 = []

    mnist = Data(balance=False)
    mnist.load_data('mnist_train.csv', 'mnist_test.csv')

    print('')
    print('Data Size: ' + str(len(mnist.train_x)))
    model = Model(input_shape=784, num_classes=10, save_path='none', verbose=False)
    accuracies1.append(model.train(mnist.train_x, mnist.train_y, mnist.test_x, mnist.test_y))

    mnist.reduce_data(percentage=0.99)

    model = Model(784, 10, 'none', verbose=False)
    model.set_loss_params(weights=mnist.get_weights())
    active = Active(mnist, model, 10, 1.00)
    accuracies1 += active.run(50, 500, 100)
    print(accuracies1)


if __name__ == '__main__':
    main()
