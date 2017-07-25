import comm
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import Ridge

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

    def Hypothesis(self, X):
        return np.dot(X, self.W) # return a m-by-1 vector

    def CostFunc(self, X, y):
        cost = 1/2 * np.average(np.square(self.Hypothesis(X) - y))
        regTerm = self.C/2 * np.sum(np.square(self.W))
        return cost + regTerm

    def Gradient(self, X, y):
        grad = 1 / X.shape[0] * np.dot(X.T, np.dot(X, self.W) - y) + self.C * self.W # n-by-1 vector
        return grad

    def UpdateGradient(self, X, Y, eta):  # eta: step size
        grad = self.Gradient(X, Y)
        newW = self.W - eta * grad
        self.W = newW

    def Fit(self, X, y):
        # deal with data first
        X_train = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        m, n = X_train.shape # m: sample size, n: feature size
        #initialize W
        self.W = np.random.rand(n) * 1e-4
        optW = 0

        # iteration: SGD algorithm
        iterCount = 1
        #previousCost = self.CostFunc(X_train, y)
        while iterCount < self.iterNum:
            index = np.random.choice(X_train.shape[0], 128)
            #eta = min(2/(self.C * (iterCount + 1)), 1)
            eta = 0.1
            self.UpdateGradient(X_train[index], y[index], eta)
            if 0 == iterCount % 100:
                #currentCost = self.CostFunc(X_train[index], y[index])
                #print("iteration: %d, cost: %f" % (iterCount, currentCost))
                print("iteration: %d, W[0-4]: %f" % (iterCount, self.W[0]))
                #if abs(previousCost - currentCost)  < self.tol:
                #    print("terminated")
                #    break
                #previousCost = currentCost
                pass
            optW += 2 * iterCount * self.W / (self.iterNum * (self.iterNum + 1))
            iterCount = iterCount + 1
        self.W = optW

    def Predict(self, X):
        X_test = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        Y_test = self.Hypothesis(X_test)
        return Y_test

    def Score(self, X, y):
        y_pred = self.Predict(X)
        return explained_variance_score(y, y_pred)


if __name__ == '__main__':
    # load data
    X_train, X_test, y_train, y_test = comm.LoadTxtData('../data/YearPredictionMSD.txt', test_size=0.05, scale=False)
    # # fit model
    # model = Model(tol=1e-4, C=1.0e0, iterNum=100000)
    # model.Fit(X_train, y_train)
    # # test
    # print("training accuracy:", model.Score(X_train, y_train))
    # print("test accuracy:", model.Score(X_test, y_test))

    # scikit learn
    clf = Ridge(alpha=1.0)
    clf.fit(X_train, y_train)
    # test
    print("\n")
    print("training accuracy:", explained_variance_score(y_train, clf.predict(X_train)))
    print("test accuracy:", explained_variance_score(y_test, clf.predict(X_test)))
