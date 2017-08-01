import comm
import math
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

PLOT_NUM = 4000

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
        self.optSolution = 0.381876662416058776372551619715522974729537963867187500

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
            self.W = self.SVRG(X_train, y, 100, int(self.iterNum/100))
        elif solver == 'SGD':
            self.W = self.SGD(X_train, y)  # SGD optimization
        elif solver == 'SAGA':
            self.W = self.SAGA(X_train, y)  # SGD optimization

        print("total cost: %.54f" % (self.CostFunc(self.W, X_train, y)))

    def SGD(self, X_train, y):
        # iteration: SGD algorithm
        optW = 0
        iterCount = 1
        previousCost = self.CostFunc(self.W, X_train, y)
        print("iteration: %d, cost: %f" % (iterCount, previousCost))
        while iterCount < self.iterNum:
            index = np.random.choice(X_train.shape[0], 128)
            #eta = min(2/(self.C * (iterCount + 1)), 1)
            eta = 0.01
            self.UpdateGradient(X_train[index], y[index], eta)
            if 0 == iterCount % 100:
                currentCost = self.CostFunc(optW, X_train, y)
                print("iteration: %d, cost: %f" % (iterCount, currentCost))
                #print("iteration: %d, W[0]: %f" % (iterCount, self.W[0]))
                #if abs(previousCost - currentCost)  < self.tol:
                #    print("terminated")
                #    break
                #previousCost = currentCost
            optW += 2 * iterCount * self.W / (self.iterNum * (self.iterNum + 1))
            iterCount = iterCount + 1

            # we need to store the cost functions so that we can plot them
            if iterCount < PLOT_NUM:
                points.append([iterCount, math.log(self.CostFunc(self.W, X_train, y) - self.optSolution, 10)])
        return optW

    def SVRG(self, X_train, Y_train, iterNum, epoch, eta=5.875e-3):
        # SVRG algorithm

        w_tilde = self.W
        for s in range(epoch):
            W = w_tilde
            n_tilde = self.Gradient(w_tilde, X_train, Y_train)

            #indices = np.random.choice(X_train.shape[0], 50)
            print("iteration: %d, cost: %.54f" % (s * iterNum, self.CostFunc(self.W, X_train, Y_train)))
            for t in range(iterNum):
                index = np.random.choice(X_train.shape[0], 1)
                deltaW = (self.Gradient(W, X_train[index], Y_train[index]) - self.Gradient(w_tilde, X_train[index], Y_train[index]) + n_tilde)
                W = W - eta * deltaW

                # we need to store the cost functions so that we can plot them
                if s * iterNum + t < PLOT_NUM:
                    points.append([s*iterNum+t, math.log(self.CostFunc(W, X_train, Y_train) - self.optSolution, 10)])

            w_tilde = W
            self.W = w_tilde
        return w_tilde

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
    # fit model
    points = []  # clear the list of costs of all iterations
    model = Model(tol=1e-4, C=1e-3, iterNum=4003)
    model.Fit(X_train, y_train, solver='SVRG')
    # test
    print("training accuracy:", model.Score(X_train, y_train))
    print("test accuracy:", model.Score(X_test, y_test))

    # scikit learn
    clf = Ridge(alpha=1e4)
    clf.fit(X_train, y_train)
    # test
    print("\n")
    print("training accuracy:", explained_variance_score(y_train, clf.predict(X_train)))
    print("test accuracy:", explained_variance_score(y_test, clf.predict(X_test)))

    # plot the convergence curve
    plt.figure("log-suboptimality of SVRG")
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1], label='SVRG')
    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    y_min = points[:, 1].min()
    y_max = points[:, 1].max()


    # fit again
    points = []
    model.Fit(X_train, y_train, solver='SGD')

    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1], label='SGD')

    x_min = min(points[:, 0].min(), x_min)
    x_max = max(points[:, 0].max(), x_max)
    y_min = min(points[:, 1].min(), y_min)
    y_max = max(points[:, 1].max(), y_max)

    plt.legend()
    plt.xlabel('#iterations')
    plt.ylabel('log-suboptimality')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
