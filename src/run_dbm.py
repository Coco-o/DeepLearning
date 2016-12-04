import models.dbm
import numpy as np
import utils

def calculate_error(model, error, X, Y):
    loss = model.loss(X, 10)
    error.append(np.sum(loss) / len(Y))
    print np.sum(loss) / len(Y)

def sample(model):
    K = 100
    T = 1000
    v = np.random.randint(2, size=(K, model.D))
    h1 = np.random.randint(2, size=(K, model.H1))
    h2 = np.random.randint(2, size=(K, model.H2))
    for t in range(T):
        v, h1, h2, cond_v = model.gibbs_sampling(v, h1, h2, K)
    np.savetxt('100_sample', v)

def main():
    Xtrain, Ytrain = utils.load_data('../data/digitstrain.txt')
    Xvalid, Yvalid = utils.load_data('../data/digitsvalid.txt')

    K = 100
    model = models.dbm.Model(784, 100, 100, K)
    N = 100
    alpha = 0.01
    T = 1000

    train_error = []
    valid_error = []
    size = len(Ytrain)
    batch = size / N
    for t in range(T):
        print '===== start training epoch ', t, ' ====='
        indx = np.split(np.arange(size), batch)
        for b in range(batch):
            model.run_dbm(Xtrain[indx[b], :], alpha)

        print 'compute loss'
        calculate_error(model, train_error, Xtrain, Ytrain)
        calculate_error(model, valid_error, Xvalid, Yvalid)

    print '===== visulaize ====='
    # visualize the learned features
    W = np.transpose(model.W1)
    sample(model)
    utils.plot_curve(train_error, valid_error, 'cross entropy error')

if __name__=='__main__':
    main()
