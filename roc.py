import comm
import math
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import random

classs Model:
    """ ROC curve
    @attributes
    tol: tolerance of stopping criteria
    C: regularization strength
    W: n coefficients vector
    results: a list that stores each epoch's sub-optimality
    """

    def __init__(self, tol=1e-4, C=1.0, iterNum=1000):
        self.tol = tol
        self.C = C
        self.iterNum = iterNum
        self.optSolution = 0
        self.results = []

    def Hypothesis(self, W, X):
        return np.dot(X, W) # return a m-by-1 vector

    def CostFunc(self, W, dual, X, y):
        m, n = X.shape
        e_pos = np.zeros_like(y)
        e_pos[np.argwhere(y == 1)] = 1
        n_pos = np.count_nonzero(e_pos)
        e_neg = np.zeros_like(y)
        e_neg[np.argwhere(y == -1)] = 1
        n_neg = np.count_nonzero(e_neg)
        a = e_pos/n_pos - e_neg/n_neg
        M = 1 / (1/n_pos + 1/n_neg)
        A = np.diag(e_pos/n_pos + e_neg/n_neg) - 1/(n_pos * n_neg) * (np.dot(e_pos, e_neg.T), np.dot(e_neg, e_pos.T)
        Ainv = np.linalg.inv(A)

        loss = 1/2 + np.dot(dual.T - a.T, np.dot(X, W) - M/2*np.sum(np.square(dual)) - 1/2*np.dot(dual.T, np.dot(Ainv - M*np.identity(m), dual))
        # compute sum(W[i] - W[j]) where i != j
        clusterInducingTerm = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                clusterInducingTerm += abs(W[i] - W[j])
        clusterInducingTerm = clusterInducingTerm / 2
        regTerm = self.C/2 * np.sum(np.square(W)) + clusterInducingTerm
        return loss + regTerm

if __name__ == '__main__':
    # load data
    X_train, X_test, y_train, y_test = comm.LoadSidoData(test_size=0.05)
    model = Model(tol=1e-4, C=1e-3, iterNum=1001)

    # a new figure
    plt.figure("sido")
    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf

    solvers = ['SAGA']
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
    plt.savefig('roc.png', dpi=96)
