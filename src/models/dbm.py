import numpy as np

def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))

class Model:
    # W1: H1 x D
    # W2: H2 x H1
    def __init__(self, D, H1, H2, K):
        self.D = D
        self.H1 = H1
        self.H2 = H2
        self.K = K
        self.W1 = np.random.normal(0, 0.1, (H1, D))
        self.W2 = np.random.normal(0, 0.1, (H2, H1))
        self.b = np.zeros(D)
        self.c1 = np.zeros(H1)
        self.c2 = np.zeros(H2)

        # initialize sampled V, Hd1, Hd2
        self.V_k = np.random.randint(2, size=(K, D))
        self.Hd1_k = np.random.randint(2, size=(K, H1))
        self.Hd2_k = np.random.randint(2, size=(K, H2))

    ########## Variational Inference ##########
    # this function returns the variational parameters mu1, mu2
    # for each sample (N) run the variational inference for 10 steps
    # V: N x D
    # mu1: N x H1
    # mu2: N x H2
    def variational_inference(self, V):
        N = V.shape[0]
        step = 10
        # random initialize Mu
        Mu1 = np.random.uniform(0, 1, (N, self.H1))
        Mu2 = np.random.uniform(0, 1, (N, self.H2))
        for s in range(step):
            Mu1 = self.q_mu1(V, Mu2, N)
            Mu2 = self.q_mu2(Mu1, N)
        return Mu1, Mu2

    def q_mu1(self, V, Mu2, N):
        return sigmoid(np.dot(V, np.transpose(self.W1)) + np.dot(Mu2, self.W2) + np.tile(self.c1, (N, 1)))

    def q_mu2(self, Mu1, N):
        return sigmoid(np.dot(Mu1, np.transpose(self.W2)) + np.tile(self.c2, (N, 1)))

    ########## Persistent CD ##########
    # this function returns the sampled K chains of V, Hd1, Hd2
    # perform 1-step Gibbs sampling for K chains
    # here V, Hd1, Hd2 are sampled parameters
    # V: K x D
    # Hd1: K x H1
    # Hd2: K x H2
    def gibbs_sampling(self, V, Hd1, Hd2, K):
        prob_h1 = np.random.uniform(0, 1, (K, self.H1))
        cond_h1 = self.conditional_h1(V, Hd2, K)
        Hd1 = (prob_h1 <= cond_h1).astype(float)

        prob_h2 = np.random.uniform(0, 1, (K, self.H2))
        cond_h2 = self.conditional_h2(Hd1, K)
        Hd2 = (prob_h2 <= cond_h2).astype(float)

        prob_v = np.random.uniform(0, 1, (K, self.D))
        cond_v = self.conditional_v(Hd1, K)
        V = (prob_v <= cond_v).astype(float)

        return V, Hd1, Hd2, cond_v

    def conditional_h1(self, V, Hd2, K):
        return sigmoid(np.dot(V, np.transpose(self.W1)) + np.dot(Hd2, self.W2) + np.tile(self.c1, (K, 1)))

    def conditional_h2(self, Hd1, K):
        return sigmoid(np.dot(Hd1, np.transpose(self.W2)) + np.tile(self.c2, (K, 1)))

    def conditional_v(self, Hd1, K):
        return sigmoid(np.dot(Hd1, self.W1) + np.tile(self.b, (K, 1)))

    ########## Deep Boltzmann Machine Learning ##########
    def run_dbm(self, V, alpha):
        N = V.shape[0]
        Mu1, Mu2 = self.variational_inference(V)
        for n in range(5):
            self.V_k, self.Hd1_k, self.Hd2_k, dummy = self.gibbs_sampling(self.V_k, self.Hd1_k, self.Hd2_k, self.K)

        # update parameters
        self.W1 += alpha * (1.0 / N * np.dot(np.transpose(Mu1), V) - 1.0 / self.K * np.dot(np.transpose(self.Hd1_k), self.V_k))
        self.W2 += alpha * (1.0 / N * np.dot(np.transpose(Mu2), Mu1) - 1.0 / self.K * np.dot(np.transpose(self.Hd2_k), self.Hd1_k))
        self.b += alpha * (1.0 / N * np.sum(V, axis=0) - 1.0 / self.K * np.sum(self.V_k, axis=0))
        self.c1 += alpha * (1.0 / N * np.sum(Mu1, axis=0) - 1.0 / self.K * np.sum(self.Hd1_k, axis=0))
        self.c2 += alpha * (1.0 / N * np.sum(Mu2, axis=0) - 1.0 / self.K * np.sum(self.Hd2_k, axis=0))

    def loss(self, V, T):
        K = V.shape[0]
        v = V
        h1 = np.random.randint(2, size=(K, self.H1))
        h2 = np.random.randint(2, size=(K, self.H2))
        for t in range(T):
            v, h1, h2, cond_v = self.gibbs_sampling(v, h1, h2, K)
        return - np.multiply(V, np.log(cond_v)) - np.multiply((1 - V), np.log(1 - cond_v))
