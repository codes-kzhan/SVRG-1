import comm
import numpy as np
import math
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

PLOT_NUM = 4000

class Model:
    """ logistic regression model
    @attributes:
    tol: tolerance of stopping criteria
    C: regularization strength
    m: sample size
    n: feature size
    k: number of classes
    W: n*k coefficient matrix
    """
    def __init__(self, tol=1e-4, C=1.0, iterNum=1000):
        self.tol = tol
        self.C = C
        self.iterNum = iterNum
        self.optSolution = 0.872841085854215270600775511411484330892562866210937500

    def Hypothesis(self, W, X):
        tmpH = np.exp(np.dot(X, W))
        denominator = np.sum(tmpH, axis=1)
        return tmpH / denominator.reshape(len(denominator), 1) # return a m-by-k matrix

    def CostFunc(self, W, X, Y):
        cost = -np.sum(np.multiply(Y, np.log(self.Hypothesis(W, X))))
        regTerm = self.C/2 * np.sum(np.square(W))
        return 1 / X.shape[0] * cost + regTerm

    def Gradient(self, W, X, Y):
        return 1 / X.shape[0] * np.dot(X.T, (self.Hypothesis(W, X) - Y)) + self.C * W  # n-by-k gradient matrix

    def UpdateGradient(self, X, Y, eta):  # eta: step size
        grad = self.Gradient(self.W, X, Y)
        newW = self.W - eta * grad
        self.W = newW

    def Fit(self, X, y, solver='SVRG'):
        # deal with data first
        X_train = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        m, n = X_train.shape # m: sample size, n: feature size
        self.classes = np.unique(y)
        self.k = self.classes.shape[0]
        #initialize W
        self.W = np.random.rand(n, self.k) * 1e-2
        # binarize labels
        self.lb = LabelBinarizer(sparse_output=False)  # @NOTE I don't know whether it should be sparse or not
        self.lb.fit(self.classes)
        Y_train = self.lb.transform(y)  # make y_train a m*k matrix
        if solver == 'SVRG':
            self.W = self.SVRG(X_train, Y_train, 100, int(self.iterNum / 100), 4.4531e-1)
        elif solver == 'SGD':
            self.W = self.SGD(X_train, Y_train)  # SGD optimization
        elif solver == 'SAGA':
            pass # @TODO
        print("total cost: %.54f" % (self.CostFunc(self.W, X_train, Y_train)))


    def SGD(self, X_train, Y_train):
        # iteration: SGD algorithm
        optW = 0
        iterCount = 1
        previousCost = self.CostFunc(self.W, X_train, Y_train)
        print("iteration: %d, cost: %f" % (iterCount, previousCost))
        while iterCount < self.iterNum:
            index = np.random.choice(X_train.shape[0], 128)
            eta = min(2/(self.C * (iterCount + 1)), 1)
            self.UpdateGradient(X_train[index], Y_train[index], eta)
            if 0 == iterCount % 50:
                currentCost = self.CostFunc(optW, X_train, Y_train)
                print("iteration: %d, cost: %f" % (iterCount, currentCost))
                if abs(previousCost - currentCost)  < self.tol:
                    print("terminated")
                    break
                previousCost = currentCost
            optW += 2 * iterCount * self.W / (self.iterNum * (self.iterNum + 1))
            iterCount = iterCount + 1

            # we need to store the cost functions so that we can plot them
            if iterCount < PLOT_NUM:
                points.append([iterCount, math.log(self.CostFunc(self.W, X_train, Y_train) - self.optSolution, 10)])

        return optW

    def SVRG(self, X_train, Y_train, iterNum, epoch, eta):
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

    def SAGA(self, X_train, Y_train, iterNum, ):


    def Predict(self, X):
        X_test = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        Y_test = self.Hypothesis(self.W, X_test)
        labels = np.zeros_like(Y_test)
        labels[np.arange(len(Y_test)), Y_test.argmax(1)] = 1
        return self.lb.inverse_transform(labels)

    def Score(self, X, y):
        y_pred = self.Predict(X)
        return accuracy_score(y, y_pred)

if __name__ == '__main__':
    # load data
    X_train, X_test, y_train, y_test = comm.LoadOpenMLData(dataset_id=150, test_size=0.05)
    model = Model(tol=1e-8, C=2.625e-3, iterNum=4003)
    # fit model

    points = []
    # clear the list of costs of all iterations
    model.Fit(X_train, y_train, solver='SVRG')
    # test
    print("training accuracy:", model.Score(X_train, y_train))
    print("test accuracy:", model.Score(X_test, y_test))

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
