import numpy as np
import matplotlib.pyplot as plt
import argparse

class Layer:
    def __init__(self, dim1, dim2, dropout=False):
        self.W = np.random.normal(0, 0.1, (dim1, dim2))
        self.b = np.zeros(dim2)
        self.dropout = dropout

        # add masks to the hidden layer with dropout set to True
        if dropout:
            self.m = np.random.randint(2, size=dim2)
            self.test_m = 0.5 * np.ones(dim2)

# Softmax layer
class Softmax(Layer):
    def forward(self, X, test=False):
        if not self.dropout:
            return np.exp(X) / np.sum(np.exp(X))
        elif test:
            return self.test_m * (np.exp(X) / np.sum(np.exp(X)))
        else:
            return self.m * (np.exp(X) / np.sum(np.exp(X)))

    def last_backward(self, f_x, y):
        dim = f_x.shape[0]
        e_y = np.zeros(dim)
        e_y[int(y)] = 1.0
        return -(e_y - f_x)

# Sigmoid layer
class Sigmoid(Layer):
    def forward(self, X, test=False):
        if not self.dropout:
            return 1.0 / (1.0 + np.exp(-X))
        elif test:
            return self.test_m * 1.0 / (1.0 + np.exp(-X))
        else:
            return self.m * 1.0 / (1.0 + np.exp(-X))

    def backward(self, X):
        g_x = self.forward(X)
        return g_x * (1 - g_x)

    def last_backward(self, f_x, y):
        return f_x - y

class Model:
    def __init__(self):
        self.layers = []
        self.A_x = []
        self.prev_dW = []
        self.prev_db = []

    def add(self, cur_layer):
        self.layers.append(cur_layer)
        self.A_x.append(np.zeros_like(cur_layer.b))
        self.prev_dW.append(np.zeros_like(cur_layer.W))
        self.prev_db.append(np.zeros_like(cur_layer.b))

    # forward propagation in training phase
    def predict(self, X):
        input = X
        for i in range(len(self.layers)):
            cur_layer = self.layers[i]
            a_x = cur_layer.b + np.dot(np.transpose(cur_layer.W), input)
            self.A_x[i] = a_x
            self.prev_dW[i].fill(0.)
            self.prev_db[i].fill(0.)
            input = cur_layer.forward(a_x)

        return input

    # model training
    def train(self, X, y, alpha, beta, lmbda):
        # forward propagetion
        f_x = self.predict(X)

        # back propagation
        da = self.layers[-1].last_backward(f_x, y)
        for i in reversed(range(0, len(self.layers))):
            db = da;
            if i == 0:
                dW = np.outer(X, da)
            else:
                a_x = self.A_x[i - 1]
                dW = np.outer(self.layers[i - 1].forward(a_x), da)
                dh = np.dot(self.layers[i].W, da)
                da = dh * self.layers[i - 1].backward(a_x)

            # add regularization and momentum
            dW += beta * self.prev_dW[i] + 2 * lmbda * self.layers[i].W
            db += beta * self.prev_db[i]
            self.prev_dW[i] = dW
            self.prev_db[i] = db

            # gradient descent
            self.layers[i].W -= alpha * dW
            self.layers[i].b -= alpha * db

    # forward propagation in testing phase
    def test(self, X):
        input = X
        for i in range(len(self.layers)):
            cur_layer = self.layers[i]
            a_x = cur_layer.b + np.dot(np.transpose(cur_layer.W), input)
            input = cur_layer.forward(a_x, True)

        return input

    # calculate the loss_function
    def loss(self, x, y):
        f_x = self.test(x)
        label = np.argmax(f_x)

        # cross entropy error and classification error
        return -np.log(f_x[int(y)]), 1.0 if label == y else 0.0

