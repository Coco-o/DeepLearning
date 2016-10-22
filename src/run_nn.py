import matplotlib.pyplot as plt
import numpy as np
import argparse
import neural_network

# run neural network model and plot error curves
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--beta')
    parser.add_argument('-a', '--alpha')
    parser.add_argument('-l', '--lmbda')
    parser.add_argument('-e', '--epoch')
    parser.add_argument('-d', '--dropout')
    parser.add_argument('-i', '--hidden')

    return parser.parse_args()

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    indx = np.arange(data.shape[0])
    np.random.shuffle(indx)
    data = data[indx]
    X = data[:, 0: 784]
    Y = data[:, 784]

    return X, Y

def visualize(W):
    for j in range(W.shape[1]):
        weight = W[:,j]
        weight = np.reshape(weight, (28, 28))
        plt.subplot(W.shape[1] / 10, 10, j + 1)
        plt.imshow(weight)
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

def calculate_error(model, cross, accuracy, X, Y):
    sum = 0
    etr = 0
    for i in range(len(Y)):
        entropy, error = model.loss(X[i, :], Y[i])
        etr += entropy
        sum += error

    cross.append(etr / len(Y))
    accuracy.append(1 - sum / len(Y))

def main():
    args = parser()
    hidden = int(args.hidden)
    alpha = float(args.alpha)
    beta = float(args.beta)
    lmbda = float(args.lmbda)
    epoch = int(args.epoch)
    dropout = True

    model = neural_network.Model(dropout)
    model.add(neural_network.Sigmoid(784, hidden))
    model.add(neural_network.Softmax(hidden, 10))

    Xtrain, Ytrain = load_data('/data/digitstrain.txt')
    Xvalid, Yvalid = load_data('/data/digitsvalid.txt')

    cross_train = []
    cross_valid = []
    accuracy_train = []
    accuracy_valid = []
    for n in range(epoch):
        print(n)
        for i in range(len(Ytrain)):
            x = Xtrain[i, :]
            y = Ytrain[i]
            model.train(x, y, alpha, beta, lmbda)

        calculate_error(model, cross_train, accuracy_train, Xtrain, Ytrain)
        calculate_error(model, cross_valid, accuracy_valid, Xvalid, Yvalid)
        print(cross_train)

    for i in range(len(model.layers) - 1):
        W = model.layers[i].W
        visualize(W)

    plot_curve(cross_train, cross_valid, 'cross entropy error')
    plot_curve(accuracy_train, accuracy_valid, 'classificatoin error')

if __name__ == "__main__":
    main()
