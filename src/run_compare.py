import matplotlib.pyplot as plt
import numpy as np
import argparse
import models.autoencoder
import models.rbm
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

    Xtrain, Ytrain = utils.load_data('../data/digitstrain.txt')
    Xvalid, Yvalid = utils.load_data('../data/digitsvalid.txt')

    # use denoising autoencoder for pretraining
    denoise_auto_model = models.autoencoder.Model(784, hidden, True, 0.5)
    denoise_auto_model.add(models.autoencoder.Sigmoid(784, hidden))
    denoise_auto_model.add(models.autoencoder.Sigmoid(hidden, 784))
    denoise_auto_model.update_W()

    for n in range(epoch):
        for i in range(Xtrain.shape[0]):
            x = Xtrain[i, :]
            denoise_auto_model.train(x, x, alpha)

    # use autoencoder for pretraining
    auto_model = models.autoencoder.Model(784, hidden)
    auto_model.add(models.autoencoder.Sigmoid(784, hidden))
    auto_model.add(models.autoencoder.Sigmoid(hidden, 784))
    auto_model.update_W()

    for n in range(epoch):
        for i in range(Xtrain.shape[0]):
            x = Xtrain[i, :]
            auto_model.train(x, x, alpha)

    # use rbm for pretraining
    rbm = models.rbm.Model(hidden, 784)

    for n in range(epoch):
        for i in range(Xtrain.shape[0]):
            x = Xtrain[i, :]
            rbm.run_rbm(x, alpha, 10)

    # use random weight
    model = models.neural_network.Model()
    model.add(models.neural_network.Sigmoid(784, hidden))
    model.add(models.neural_network.Softmax(hidden, 10))

    cross_valid_random = []
    accuracy_valid_random = []
    for n in range(epoch):
        print(n)
        for i in range(len(Ytrain)):
            x = Xtrain[i, :]
            y = Ytrain[i]
            model.train(x, y, alpha, beta, lmbda)

        calculate_error(model, cross_valid_random, accuracy_valid_random, Xvalid, Yvalid)
        print(cross_valid_random[-1])

    # use the dennoising auto pre-trained weights
    model = models.neural_network.Model()
    model.add(models.neural_network.Sigmoid(784, hidden))
    model.add(models.neural_network.Softmax(hidden, 10))
    model.layers[0].W = denoise_auto_model.W

    cross_valid_denoise = []
    accuracy_valid_denoise = []
    for n in range(epoch):
        print(n)
        for i in range(len(Ytrain)):
            x = Xtrain[i, :]
            y = Ytrain[i]
            model.train(x, y, alpha, beta, lmbda)

        calculate_error(model, cross_valid_denoise, accuracy_valid_denoise, Xvalid, Yvalid)
        print(cross_valid_denoise[-1])

    # use the dennoising auto pre-trained weights
    model = models.neural_network.Model()
    model.add(models.neural_network.Sigmoid(784, hidden))
    model.add(models.neural_network.Softmax(hidden, 10))
    model.layers[0].W = auto_model.W

    cross_valid_auto = []
    accuracy_valid_auto = []
    for n in range(epoch):
        print(n)
        for i in range(len(Ytrain)):
            x = Xtrain[i, :]
            y = Ytrain[i]
            model.train(x, y, alpha, beta, lmbda)

        calculate_error(model, cross_valid_auto, accuracy_valid_auto, Xvalid, Yvalid)
        print(cross_valid_auto[-1])

    # use the dennoising auto pre-trained weights
    model = models.neural_network.Model()
    model.add(models.neural_network.Sigmoid(784, hidden))
    model.add(models.neural_network.Softmax(hidden, 10))
    model.layers[0].W = np.transpose(rbm.W)

    cross_valid_rbm = []
    accuracy_valid_rbm = []
    for n in range(epoch):
        print(n)
        for i in range(len(Ytrain)):
            x = Xtrain[i, :]
            y = Ytrain[i]
            model.train(x, y, alpha, beta, lmbda)

        calculate_error(model, cross_valid_rbm, accuracy_valid_rbm, Xvalid, Yvalid)
        print(cross_valid_rbm[-1])

    x = np.array(range(epoch))
    plt.hold(True)
    plt.plot(x, accuracy_valid_random, label='randomized')
    plt.plot(x, accuracy_valid_auto, label='autoencoder')
    plt.plot(x, accuracy_valid_denoise, label='denoising')
    plt.plot(x, accuracy_valid_rbm, label='RBM')
    plt.xlabel('Epoch')
    plt.ylabel('classification error')
    plt.legend(loc='upper center', shadow=True)
    plt.show()

if __name__ == "__main__":
    main()
