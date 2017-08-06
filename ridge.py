import comm
import math
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import random

class Model:
    """ ridge regression model
    @attributes:
    tol: tolerance of stopping criteria
    C: regularization strength
    m: sample size
    n: feature size
    k: number of classes
    W: n coefficients vector
    """
    def __init__(self, tol=1e-4, C=1.0, iterNum=1000):
        self.tol = tol
        self.C = C
        self.iterNum = iterNum
        self.optSolution = 0.381876662416058720861400388457695953547954559326171875
        self.results = []

    def Hypothesis(self, W, X):
        return np.dot(X, W) # return a m-by-1 vector

    def CostFunc(self, W, X, y):
        cost = 1/2 * np.average(np.square(self.Hypothesis(W, X) - y))
        regTerm = self.C/2 * np.sum(np.square(W))
        return cost + regTerm

    def Gradient(self, W, X, y):
        # @NOTE X MUST an m*n matrix, where m denotes #samples, n denotes #features
        m, n = X.shape # m: sample size, n: feature size
        return np.dot(1/m *  X.T, np.dot(X, W) - y) + self.C * W # n-by-1 vector

    def PrintCost(self, W, X, y, numEpoch):
        currentCost = self.CostFunc(W, X, y)
        print("epoch: %2d, cost: %.16f" % (numEpoch, currentCost))
        # we need to store the cost functions so that we can plot them
        self.results.append([numEpoch, math.log(currentCost - self.optSolution, 10)])

    def Fit(self, X, y, solver='SVRG'):
        # deal with data first
        X_train = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        m, n = X_train.shape # m: sample size, n: feature size
        self.results.clear() # we need to store each epoch's cost so that we can plot them
        #initialize W
        self.W = np.random.rand(n) * 1e-4

        # find optimal W
        if solver == 'SVRG':
            self.W = self.SVRG(X_train, y, m, int(self.iterNum/m))
        elif solver == 'SGD':
            self.W = self.SGD(X_train, y)  # SGD optimization
        elif solver == 'SAGA':
            self.W = self.SAGA(X_train, y)  # SGD optimization
        elif solver == 'WOSVRG':
            self.W = self.WOSVRG(X_train, y, m, int(self.iterNum/m))

        print("total cost: %.16f" % (self.CostFunc(self.W, X_train, y)))

    def SGD(self, X_train, y_train):
        # iteration: SGD algorithm
        W = self.W
        optW = 0
        m, n = X_train.shape # m: sample size, n: feature size
        batchSize = 10
        for t in range(self.iterNum):
            index = np.random.choice(m, batchSize) # minibatch size: 10
            eta = min(2/(self.C * (t + 2)), 0.01)
            W = W - eta * (np.dot(1/batchSize * X_train[index].T, np.dot(X_train[index], W) - y_train[index]) + self.C * W)
            optW += 2 * (t + 1) * W / (self.iterNum * (self.iterNum + 1))

            # print and plot
            if 0 == t % m:
                self.PrintCost(W, X_train, y_train, int(t/m))

        return optW

    def SVRG(self, X_train, y_train, iterNum, epoch, eta=5.875e-4):
        # SVRG algorithm

        m, n = X_train.shape # m: sample size, n: feature size
        w_tilde = self.W
        for s in range(epoch):
            # print and plot
            self.PrintCost(w_tilde, X_train, y_train, s)

            W = w_tilde
            n_tilde = np.dot(1 / m *  X_train.T, np.dot(X_train, w_tilde) - y_train) # n-by-1 vector

            for t in range(iterNum):
                index = np.random.choice(X_train.shape[0], 1)
                deltaW = np.dot(X_train[index], W - w_tilde) * X_train[index].T.reshape([n]) + self.C * W + n_tilde
                W = W - eta * deltaW

            w_tilde = W

        # print and plot the last epoch
        self.PrintCost(w_tilde, X_train, y_train, s + 1)

        return w_tilde

    def WOSVRG(self, X_train, y_train, iterNum, epoch, eta=5.875e-4):
        # Without-Replacement SVRG algorithm

        m, n = X_train.shape # m: sample size, n: feature size
        w_tilde = self.W
        for s in range(epoch):
            # print and plot
            self.PrintCost(w_tilde, X_train, y_train, s)

            W = w_tilde
            n_tilde = np.dot(1 / m *  X_train.T, np.dot(X_train, w_tilde) - y_train) # n-by-1 vector

            for t in range(iterNum):
                index = np.array([t])
                #accelerated gradient computation
                deltaW = np.dot(X_train[index], W - w_tilde) * X_train[index].T.reshape([n]) + self.C * W + n_tilde
                W = W - eta * deltaW

            w_tilde = W

        # print and plot the last epoch
        self.PrintCost(w_tilde, X_train, y_train, s + 1)

        return w_tilde

    def SAGA(self, X_train, y_train, gamma=1.0e-4):
        W = self.W
        m, n = X_train.shape # m: sample size, n: feature size

        # initialize gradients
        gradients = np.zeros([m, n])
        for i in range(X_train.shape[0]):
            gradients[i] = self.Gradient(self.W, X_train[[i]], y_train[[i]])
        # # initialize gradients
        # gradients = np.multiply((np.dot(X_train, W) - y_train).reshape([m, 1]), X_train) + self.C * W
        sum_gradients = np.sum(gradients, axis=0)
        for t in range(self.iterNum):
            # pick an index uniformly at random
            index = np.random.choice(X_train.shape[0], 1)
            index_scalar = index[0]
            # update W
            new_grad = self.Gradient(W, X_train[index], y_train[index])
            W = W - gamma * (new_grad - gradients[index_scalar] + sum_gradients/m)
            sum_gradients = sum_gradients - gradients[index_scalar] + new_grad
            gradients[index_scalar] = new_grad

            # print and plot
            if 0 == t % m:
                self.PrintCost(W, X_train, y_train, int(t/m))

        return W

    def Predict(self, X):
        X_test = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        Y_test = self.Hypothesis(self.W, X_test)
        return Y_test

    def Score(self, X, y):
        y_pred = self.Predict(X)
        return explained_variance_score(y, y_pred)


if __name__ == '__main__':
    # load data
    X_train, X_test, y_train, y_test = comm.LoadTxtData('../data/YearPredictionMSD.txt', test_size=0.05, scale=True)
    model = Model(tol=1e-4, C=1e-3, iterNum=X_train.shape[0] * 20 + 1)

    # a new figure
    plt.figure("Convergence rates of SGD, SVRG, SAGA and WOSVRG")
    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf

    #solvers = ['SGD', 'SVRG', 'SAGA', 'WOSVRG']
    #solvers = ['WOSVRG', 'SAGA']
    solvers = ['SAGA']
    #solvers = ['SVRG']
    for solver in solvers:
        # fit model
        print("\nFitting data with %s algorithm..." % solver)
        model.Fit(X_train, y_train, solver=solver)
        # test
        print("training accuracy:", model.Score(X_train, y_train))
        print("test accuracy:", model.Score(X_test, y_test))

        results = np.array(model.results)
        plt.plot(results[:, 0], results[:, 1], label=solver)

        x_min = min(results[:, 0].min(), x_min)
        x_max = max(results[:, 0].max(), x_max)
        y_min = min(results[:, 1].min(), y_min)
        y_max = max(results[:, 1].max(), y_max)

    plt.legend()
    plt.xlabel('effective pass')
    plt.ylabel('log-suboptimality')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
