import matplotlib.pyplot as plt
import numpy as np
import argparse
'''
beta: parameter for momentum
alpha: learning rate
lmbda: parameter for L2 regularization
'''
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--beta')
    parser.add_argument('-a', '--alpha')
    parser.add_argument('-l', '--lmbda')
    parser.add_argument('-e', '--epoch')
    parser.add_argument('-i', '--hidden')

    return parser.parse_args()

def load_data(filename, binarized=True):
    data = np.loadtxt(filename, delimiter=',')
    indx = np.arange(data.shape[0])
    np.random.shuffle(indx)
    data = data[indx]
    X = data[:, 0: 784]
    if binarized:
        X[X > 0.5] = 1
        X[X < 0.5] = 0
    Y = data[:, 784]

    return X, Y

def visualize(W):
    for j in range(W.shape[1]):
        weight = W[:,j]
        weight = np.reshape(weight, (28, 28))
        plt.subplot(W.shape[1] / 10, 10, j + 1)
        plt.imshow(weight, cmap='Greys_r')
        plt.axis('off')
    plt.show()

def plot_curve(error1, error2, message):
    x = np.arange(0, len(error1))
    plt.plot(x, error1)
    plt.hold(True)
    plt.plot(x, error2)
    plt.xlabel('Epoch')
    plt.ylabel(message)
    plt.show()


