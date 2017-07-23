import comm
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

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

    def Hypothesis(self, X):
        tmpH = np.exp(np.dot(X, self.W))
        denominator = np.sum(tmpH, axis=1)
        return tmpH / denominator.reshape(len(denominator), 1) # return a m-by-k matrix

    def CostFunc(self, X, Y):
        cost = -np.sum(np.multiply(Y, np.log(self.Hypothesis(X))))
        regTerm = self.C/2 * np.sum(LA.norm(self.W, axis=0))
        return 1 / X.shape[0] * (cost + regTerm)

    def Gradient(self, X, Y):
        return -(1 / X.shape[0]) * np.dot(X.T, (Y - self.Hypothesis(X))) + self.C * self.W  # n-by-k gradient matrix

    def UpdateGradient(self, X, Y, eta):  # eta: step size
        grad = self.Gradient(X, Y)
        newW = self.W - eta * grad
        self.W = newW

    def Fit(self, X, y):
        # deal with data first
        X_train = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        m, n = X_train.shape # m: sample size, n: feature size
        self.classes = np.unique(y)
        self.k = self.classes.shape[0]
        #initialize W
        self.W = np.random.rand(n, self.k) * 1e-2
        optW = 0
        # binarize labels
        self.lb = LabelBinarizer(sparse_output=False)  # @NOTE I don't know whether it should be sparse or not
        self.lb.fit(self.classes)
        Y_train = self.lb.transform(y)  # make y_train a m*k matrix

        # iteration: SGD algorithm
        iterCount = 1
        previousCost = self.CostFunc(X_train, Y_train)
        while iterCount < self.iterNum:
            index = np.random.choice(X_train.shape[0], 128)
            eta = min(2/(self.C * (iterCount + 1)), 1)
            self.UpdateGradient(X_train[index], Y_train[index], eta)
            if 0 == iterCount % 100:
                currentCost = self.CostFunc(X_train[index], Y_train[index])
                print("iteration: %d, cost: %f" % (iterCount, currentCost))
                #if abs(previousCost - currentCost)  < self.tol:
                #    print("terminated")
                #    break
                previousCost = currentCost
            optW += 2 * iterCount * self.W / (self.iterNum * (self.iterNum + 1))
            iterCount = iterCount + 1
        self.W = optW

    def Predict(self, X):
        X_test = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        Y_test = self.Hypothesis(X_test)
        labels = np.zeros_like(Y_test)
        labels[np.arange(len(Y_test)), Y_test.argmax(1)] = 1
        return self.lb.inverse_transform(labels)

    def Score(self, X, y):
        y_pred = self.Predict(X)
        return accuracy_score(y, y_pred)

if __name__ == '__main__':
    # load data
    X_train, X_test, y_train, y_test = comm.LoadData(dataset_id=150, test_size=0.05)
    # fit model
    model = Model(tol=1e-4, C=2.625e-3, iterNum=100000)
    model.Fit(X_train, y_train)
    # test
    #print("training accuracy:", model.Score(X_train, y_train))
    print("test accuracy:", model.Score(X_test, y_test))
