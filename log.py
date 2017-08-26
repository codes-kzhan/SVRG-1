import comm
import math
import numpy as np
from sklearn.metrics import accuracy_score
from numpy import linalg as LA
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import pickle

class Model:
    """ ridge regression model
    @attributes:
    tol: tolerance of stopping criteria
    C: regularization strength
    W: n coefficients vector
    results: a list that stores each epoch's sub-optimality
    """
    def __init__(self, tol=1e-4, C=1.0, iterNum=1000):
        self.tol = tol
        self.C = C
        self.iterNum = iterNum
        self.results = []

    def Hypothesis(self, W, X):
        tmpH = np.exp(np.dot(X, W/2)) # return a m-by-1 vector
        denominator = tmpH + 1/tmpH
        return tmpH / denominator


    def CostFunc(self, W, X, y):
        m, n = X.shape # m: sample size, n: feature size
        cost = np.average(np.log(1 + np.exp(np.multiply(np.dot(X, W), -y))))
        regTerm = self.C/2 * np.sum(np.square(W))
        return cost + regTerm

    def Gradient(self, W, X, y):
        # @NOTE X MUST an m*n matrix, where m denotes #samples, n denotes #features
        m, n = X.shape # m: sample size, n: feature size
        tmpExp = np.exp(np.multiply(np.dot(X, W), -y))
        return np.average(np.divide(-(y*tmpExp).reshape([m, 1]) * X, 1 + tmpExp.reshape([m, 1])), axis=0) # n-by-1 vector

    def PrintCost(self, W, X, y, numEpoch):
        currentCost = self.CostFunc(W, X, y)
        # we need to store the cost functions so that we can plot them
        if currentCost <= self.optSolution:
            print('\nOops, the new opt solution is: %.50f' % currentCost)
            pickle.dump(W, open('../data/logistic_rcv1.p', 'wb'))
            return
        logSuboptimality = math.log(currentCost - self.optSolution, 10)
        self.results.append([numEpoch, logSuboptimality])
        print("epoch: %2d, cost: %f" % (numEpoch, logSuboptimality))

    def Fit(self, X, y, solver='SVRG'):

        # deal with data first
        X_train = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        m, n = X_train.shape # m: sample size, n: feature size

        # first, we compute the optimal solution
        self.optW = pickle.load(open('../data/logistic_rcv1.p', 'rb'))
        self.optSolution = self.CostFunc(self.optW, X_train, y)

        self.results.clear() # we need to store each epoch's cost so that we can plot them
        #initialize W
        #self.W = np.random.rand(n) * 1e-4
        self.W = np.zeros(n)

        # find optimal W
        if solver == 'SVRG':
            self.W = self.SVRG(X_train, y, m, int(self.iterNum/m))
            pickle.dump(self.W, open('../data/logistic_rcv1.p', 'wb'))
        elif solver == 'SGD':
            self.W = self.SGD(X_train, y)
        elif solver == 'SAGA':
            self.W = self.SAGA(X_train, y)
        elif solver == 'WOSVRG':
            self.W = self.WOSVRG(X_train, y, m, int(self.iterNum/m))
        elif solver == 'RSSAGA':
            self.W = self.RSSAGA(X_train, y)

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
            W = W - eta * (self.Gradient(W, X_train[index], y_train[index]) + self.C * W)
            optW += 2 * (t + 1) * W / (self.iterNum * (self.iterNum + 1))

            # print and plot
            if 0 == t % m:
                self.PrintCost(W, X_train, y_train, int(t/m))

        return optW

    def SVRG(self, X_train, y_train, iterNum, epoch, eta=4.95e-1):
        # SVRG algorithm

        m, n = X_train.shape # m: sample size, n: feature size
        w_tilde = self.W
        for s in range(epoch):
            # print and plot
            self.PrintCost(w_tilde, X_train, y_train, s)

            W = w_tilde

            n_tilde = self.Gradient(W, X_train, y_train)

            for t in range(iterNum):
                index = np.random.choice(m, 1)
                deltaW = self.Gradient(W, X_train[index],y_train[index]) - self.Gradient(w_tilde, X_train[index], y_train[index])+ self.C * W + n_tilde
                W = W - eta * deltaW

            w_tilde = W

        # print and plot the last epoch
        self.PrintCost(w_tilde, X_train, y_train, s + 1)

        return w_tilde

    def WOSVRG(self, X_train, y_train, iterNum, epoch, eta=4.95e-1):
        # Without-Replacement SVRG algorithm

        m, n = X_train.shape # m: sample size, n: feature size
        w_tilde = self.W
        for s in range(epoch):
            # print and plot
            self.PrintCost(w_tilde, X_train, y_train, s)

            W = w_tilde
            n_tilde = self.Gradient(W, X_train, y_train)

            for t in range(iterNum):
                index = np.array([t])
                #accelerated gradient computation
                deltaW = self.Gradient(W, X_train[index],y_train[index]) - self.Gradient(w_tilde, X_train[index], y_train[index])+ self.C * W + n_tilde
                W = W - eta * deltaW

            w_tilde = W

        # print and plot the last epoch
        self.PrintCost(w_tilde, X_train, y_train, s + 1)

        return w_tilde

    def SAGA(self, X_train, y_train, gamma=4.25e-1):
        W = self.W
        m, n = X_train.shape # m: sample size, n: feature size

        # initialize gradients
        tmpExp = np.exp(np.multiply(np.dot(X_train, W), -y_train))
        gradients = np.divide(-(y_train*tmpExp).reshape([m, 1]) * X_train, 1 + tmpExp.reshape([m, 1])) + self.C * W
        sum_gradients = np.sum(gradients, axis=0)
        # gradients = np.zeros([m, n])
        # sum_gradients = np.zeros(n)

        # stochastic iteration
        for t in range(self.iterNum):
            # print and plot
            if 0 == t % m:
                self.PrintCost(W, X_train, y_train, int(t/m))

            # pick an index uniformly at random
            index = np.random.choice(X_train.shape[0], 1)
            index_scalar = index[0]

            # update W
            new_grad = self.Gradient(W, X_train[index], y_train[index]) + self.C * W
            W = W - gamma * (new_grad - gradients[index_scalar] + sum_gradients/m)
            sum_gradients = sum_gradients - gradients[index_scalar] + new_grad
            gradients[index_scalar] = new_grad


        return W

    def RSSAGA(self, X_train, y_train, gamma=9.25e-2):
        W = self.W
        m, n = X_train.shape # m: sample size, n: feature size

        # initialize gradients
        gradients = np.zeros([m, n])
        sum_gradients = np.zeros(n)

        for t in range(self.iterNum):
            index_scalar = t % m

            if index_scalar == 0:
                # print and plot
                self.PrintCost(W, X_train, y_train, int(t/m))
                # reshuffle data
                perm = np.random.permutation(m)
                X_train = X_train[perm]
                y_train = y_train[perm]

            index = np.array([index_scalar])
            # update W
            new_grad = self.Gradient(W, X_train[index], y_train[index]) + self.C * W
            W = W - gamma * (new_grad - gradients[index_scalar] + sum_gradients/m)
            sum_gradients = sum_gradients - gradients[index_scalar] + new_grad
            gradients[index_scalar] = new_grad

        return W

    def Predict(self, X):
        X_test = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        Y_test = self.Hypothesis(self.W, X_test)
        labels = np.ones_like(Y_test)
        labels[np.argwhere(Y_test < 0.5)] = -1
        return labels

    def Score(self, X, y):
        y_pred = self.Predict(X)
        return accuracy_score(y, y_pred)

if __name__ == '__main__':
    # load data
    X_train, X_test, y_train, y_test = comm.LoadRCV1BinaryData()
    model = Model(tol=1e-4, C=1e-4, iterNum=X_train.shape[0] * 20 + 1)

    # a new figure
    plt.figure('logistic regression with l2-norm')
    plt.title('rcv1')
    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf

    #solvers = ['RSSAGA']
    #solvers = ['SGD', 'SVRG', 'SAGA', 'WOSVRG']
    #solvers = ['WOSVRG', 'SAGA']
    solvers = ['RSSAGA', 'WOSVRG', 'SAGA', 'SVRG']
    #solvers = ['RSSAGA']
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
    plt.savefig('log_rcv1.png', dpi=96)
    # plt.show()
