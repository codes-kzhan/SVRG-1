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

    def Hypothesis(self, W, X):
        return np.dot(X, W) # return a m-by-1 vector

    def CostFunc(self, W, X, y):
        cost = 1/2 * np.average(np.square(self.Hypothesis(W, X) - y))
        regTerm = self.C/2 * np.sum(np.square(W))
        return cost + regTerm

    def Gradient(self, W, X, y):
        grad = np.dot(1 / X.shape[0] *  X.T, np.dot(X, W) - y) + self.C * W # n-by-1 vector
        return grad

    def UpdateGradient(self, X, Y, eta):  # eta: step size
        grad = self.Gradient(self.W, X, Y)
        newW = self.W - eta * grad
        self.W = newW

    def Fit(self, X, y, solver='SVRG'):
        # deal with data first
        X_train = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        m, n = X_train.shape # m: sample size, n: feature size
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

    def SGD(self, X_train, y):
        # iteration: SGD algorithm
        optW = 0
        for iterCount in range(self.iterNum):
            index = np.random.choice(X_train.shape[0], 10)
            eta = min(2/(self.C * (iterCount + 2)), 0.01)
            #eta = 0.01
            self.UpdateGradient(X_train[index], y[index], eta)
            if 0 == iterCount % X_train.shape[0]:
                currentCost = self.CostFunc(self.W, X_train, y)
                print("epoch: %2d, cost: %.16f" % (int(iterCount/X_train.shape[0]), currentCost))
                # we need to store the cost functions so that we can plot them
                points.append([int(iterCount/X_train.shape[0]), math.log(currentCost - self.optSolution, 10)])

            optW += 2 * (iterCount + 1) * self.W / (self.iterNum * (self.iterNum + 1))
            iterCount = iterCount + 1

        return optW

    def SVRG(self, X_train, Y_train, iterNum, epoch, eta=5.875e-4):
        # SVRG algorithm

        w_tilde = self.W
        for s in range(epoch):

            cost = self.CostFunc(self.W, X_train, Y_train)
            print("epoch: %2d, cost: %.16f" % (s, cost))
            # we need to store the cost functions so that we can plot them
            try:
                logSuboptimality = math.log(cost - self.optSolution, 10)
            except:
                print("cost: %.54f\nopt: %.54f" % (cost, self.optSolution))
            points.append([s, logSuboptimality])

            W = w_tilde
            n_tilde = self.Gradient(w_tilde, X_train, Y_train)

            for t in range(iterNum):
                index = np.random.choice(X_train.shape[0], 1)
                deltaW = (self.Gradient(W, X_train[index], Y_train[index]) - self.Gradient(w_tilde, X_train[index], Y_train[index]) + n_tilde)
                W = W - eta * deltaW

            w_tilde = W
            self.W = w_tilde

        # last pixel
        cost = self.CostFunc(self.W, X_train, Y_train)
        print("epoch: %2d, cost: %.16f" % (s + 1, cost))
        # we need to store the cost functions so that we can plot them
        try:
            logSuboptimality = math.log(cost - self.optSolution, 10)
        except:
            print("cost: %.54f\nopt: %.54f" % (cost, self.optSolution))
        points.append([s + 1, logSuboptimality])

        return w_tilde

    def WOSVRG(self, X_train, Y_train, iterNum, epoch, eta=5.875e-4):
        # SVRG algorithm

        w_tilde = self.W

        perm = np.array(list(range(X_train.shape[0])))  # @NOTE iterNum must be #samples
        #np.random.shuffle(perm)

        for s in range(epoch):
            W = w_tilde
            n_tilde = self.Gradient(w_tilde, X_train, Y_train)

            #indices = np.random.choice(X_train.shape[0], 50)
            cost = self.CostFunc(self.W, X_train, Y_train)
            print("epoch: %2d, cost: %.16f" % (s, cost))

            # we need to store the cost functions so that we can plot them
            points.append([s, math.log(cost - self.optSolution, 10)])

            for t in range(iterNum):
                #index = perm[s * iterNum + t]
                index = np.array([perm[t]])
                #index = t
                #index = np.random.choice(X_train.shape[0], 1)
                deltaW = (self.Gradient(W, X_train[index], Y_train[index]) - self.Gradient(w_tilde, X_train[index], Y_train[index]) + n_tilde)
                W = W - eta * deltaW

            w_tilde = W
            self.W = w_tilde

        # last pixel
        cost = self.CostFunc(self.W, X_train, Y_train)
        print("epoch: %2d, cost: %.16f" % (s + 1, cost))
        # we need to store the cost functions so that we can plot them
        try:
            logSuboptimality = math.log(cost - self.optSolution, 10)
        except:
            print("cost: %.54f\nopt: %.54f" % (cost, self.optSolution))
        points.append([s + 1, logSuboptimality])

        return w_tilde


    def SAGA(self, X_train, Y_train, gamma=2.5e-4):
        W = self.W
        # initialize gradients
        gradients = np.zeros([X_train.shape[0], X_train.shape[1]])
        for i in range(X_train.shape[0]):
            gradients[i] = self.Gradient(self.W, X_train[[i]], Y_train[[i]])
        sum_gradients = np.sum(gradients, axis=0)
        for t in range(self.iterNum):
            # pick an index uniformly at random
            index = np.random.choice(X_train.shape[0], 1)
            index_scalar = index[0]
            # update W
            new_grad = self.Gradient(W, X_train[index], Y_train[index])
            #W = (1 - gamma * self.C) * W - gamma * (new_grad - gradients[index_scalar] + sum_gradients/X_train.shape[0])
            W = W - gamma * (new_grad - gradients[index_scalar] + sum_gradients/X_train.shape[0])
            #W = 1 / (self.C * gamma + 1) * W_prime
            sum_gradients = sum_gradients - gradients[index_scalar] + new_grad
            gradients[index_scalar] = new_grad

            if 0 == t % X_train.shape[0]:
                currentCost = self.CostFunc(W, X_train, Y_train)
                print("epoch: %2d, cost: %.16f" % (int(t/X_train.shape[0]), currentCost))

                # we need to store the cost functions so that we can plot them
                points.append([int(t/X_train.shape[0]), math.log(currentCost - self.optSolution, 10)])

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
    #solvers = ['SVRG', 'WOSVRG']
    solvers = ['SAGA']
    #solvers = ['SVRG']
    for solver in solvers:
        # fit model
        points = []  # clear the list of costs of all iterations
        print("\nFitting data with %s algorithm..." % solver)
        model.Fit(X_train, y_train, solver=solver)
        # test
        print("training accuracy:", model.Score(X_train, y_train))
        print("test accuracy:", model.Score(X_test, y_test))

        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], label=solver)

        x_min = min(points[:, 0].min(), x_min)
        x_max = max(points[:, 0].max(), x_max)
        y_min = min(points[:, 1].min(), y_min)
        y_max = max(points[:, 1].max(), y_max)

    plt.legend()
    plt.xlabel('effect pass')
    plt.ylabel('log-suboptimality')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
