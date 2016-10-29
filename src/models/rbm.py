import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))

class Model:
    # n denotes the number of hidden units
    # m denotes the number of visible units
    # k denotes the number of contrastive divergence steps
    def __init__(self, n, m):
        self.m = m
        self.n = n
        self.W = np.random.normal(0, 0.01, (n, m))
        self.b = np.zeros(m)
        self.c = np.zeros(n)

    # c: n x 1
    # W: n x m
    # v: m x 1
    # return the conditional distribution p(h | v)
    def conditional_h(self, v):
        return sigmoid(self.c + np.dot(self.W, v))

    # b: m x 1
    # h: n x 1
    # W: n x m
    # return the conditional distribution p(v | h)
    def conditional_v(self, h):
        return sigmoid(self.b + np.dot(np.transpose(self.W), h))

    def gibbs_sampling(self, v, k):
        for t in range(k):
            prob_h = np.random.uniform(0, 1, self.n)
            cond_h = self.conditional_h(v)
            h = np.array([0 if prob_h[i] > cond_h[i] else 1 for i in range(self.n)])

            prob_v = np.random.uniform(0, 1, self.m)
            cond_v = self.conditional_v(h)
            v = np.array([0 if prob_v[i] > cond_v[i] else 1 for i in range(self.m)])

        return v, cond_v

    def run_rbm(self, v, alpha, k):
        vk, cond_v = self.gibbs_sampling(v, k)
        cond_h_v = self.conditional_h(v)
        cond_h_vk = self.conditional_h(vk)
        self.W += alpha * (np.outer(cond_h_v, v) - np.outer(cond_h_vk, vk))
        self.b += alpha * (v - vk)
        self.c += alpha * (cond_h_v - cond_h_vk)

    def loss(self, v, k):
        vk, cond_v = self.gibbs_sampling(v, k)
        return - np.dot(v, np.log(cond_v)) - np.dot((1 - v), np.log(1 - cond_v))


