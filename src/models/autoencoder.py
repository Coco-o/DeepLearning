import matplotlib.pyplot as plt
import numpy as np
import argparse

class Layer:
    def __init__(self, dim1, dim2):
        self.W = []
        self.b = np.zeros(dim2)

# Softmax layer
class Softmax(Layer):
    def forward(self, X):
        return np.exp(X) / np.sum(np.exp(X))

    def last_backward(self, f_x, y):
        dim = f_x.shape[0]
        e_y = np.zeros(dim)
        e_y[int(y)] = 1.0
        return -(e_y - f_x)

# Sigmoid layer
class Sigmoid(Layer):
    def forward(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def backward(self, X):
        g_x = self.forward(X)
        return g_x * (1 - g_x)

    def last_backward(self, f_x, y):
        return f_x - y

class Model:
    def __init__(self, dim1, dim2, denoising=False, prob=0.0):
        self.layers = []
        self.A_x = []
        self.W = np.random.normal(0, 0.1, (dim1, dim2))
        self.denoising = denoising
        self.prob = prob

    def add(self, cur_layer):
        self.layers.append(cur_layer)
        self.A_x.append(np.zeros_like(cur_layer.b))

    def update_W(self):
        self.layers[0].W = self.W
        self.layers[1].W = np.transpose(self.W)

    # forward propagation in training phase
    def predict(self, X):
        input = X
        for i in range(len(self.layers)):
            cur_layer = self.layers[i]
            a_x = cur_layer.b + np.dot(np.transpose(cur_layer.W), input)
            self.A_x[i] = a_x
            input = cur_layer.forward(a_x)
        return input

    # model training
    def train(self, X, y, alpha):
        # add random noise to input
        if self.denoising:
            noise = np.random.uniform(0, 1, size=X.shape[0])
            noise[noise > self.prob] = 1
            noise[noise < self.prob] = 0
            X = np.multiply(X, noise)

        f_x = self.predict(X)
        da = self.layers[-1].last_backward(f_x, y)
        for i in reversed(range(0, len(self.layers))):
            db = da;
            if i == 0:
                dW = np.outer(X, da)
            else:
                a_x = self.A_x[i - 1]
                dW = np.transpose(np.outer(self.layers[i - 1].forward(a_x), da))
                dh = np.dot(self.layers[i].W, da)
                da = dh * self.layers[i - 1].backward(a_x)

            self.layers[i].b -= alpha * db
            self.W -= alpha * dW
            self.update_W()

    # forward propagation in testing phase
    def test(self, X):
        input = X
        for i in range(len(self.layers)):
            cur_layer = self.layers[i]
            a_x = cur_layer.b + np.dot(np.transpose(cur_layer.W), input)
            input = cur_layer.forward(a_x)
        return input

    def loss(self, x):
        f_x = self.test(x)
        return -sum(np.multiply(np.log(f_x), x) + np.multiply(np.log(1 - f_x), 1 - x))
