from Data import Data
from Model import Model


def main():
    mnist = Data()
    mnist.load_data('mnist_train.csv', 'mnist_test.csv')

    model = Model(784, 10, 'none')
    model.set_loss_params(weights=mnist.get_weights())
    model.train(mnist.train_x, mnist.train_y, mnist.test_x, mnist.test_y, confuse=True)


if __name__ == '__main__':
    main()
