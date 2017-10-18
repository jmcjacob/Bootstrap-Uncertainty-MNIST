from Data import Data
from Model import Model
from Active import Active


def main():
    mnist = Data(balance=False)
    mnist.load_data('mnist_train.csv', 'mnist_test.csv')

    print('')
    print('Data Size: ' + str(len(mnist.train_x)))
    model = Model(input_shape=784, num_classes=10, save_path='none', verbose=False)
    model.set_loss_params(weights=mnist.get_weights())
    model.train(mnist.train_x, mnist.train_y, mnist.test_x, mnist.test_y)

    mnist.reduce_data(percentage=0.99)

    active = Active(mnist, Model(784, 10, 'none', verbose=False), 10, 0.85)
    print(active.run(50, 500, 100))


if __name__ == '__main__':
    main()
