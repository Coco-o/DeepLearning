import matplotlib.pyplot as plt
import numpy as np
import argparse
import models.neural_network
import utils

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
    args = utils.parser()
    hidden = int(args.hidden)
    alpha = float(args.alpha)
    beta = float(args.beta)
    lmbda = float(args.lmbda)
    epoch = int(args.epoch)

    model = models.neural_network.Model()
    model.add(models.neural_network.Sigmoid(784, hidden))
    model.add(models.neural_network.Softmax(hidden, 10))

    Xtrain, Ytrain = utils.load_data('../data/digitstrain.txt', False)
    Xvalid, Yvalid = utils.load_data('../data/digitsvalid.txt', False)

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
        print(cross_train[-1])

    for i in range(len(model.layers) - 1):
        W = model.layers[i].W
        utils.visualize(W)

    utils.plot_curve(cross_train, cross_valid, 'cross entropy error')
    utils.plot_curve(accuracy_train, accuracy_valid, 'classificatoin error')

if __name__ == "__main__":
    main()
