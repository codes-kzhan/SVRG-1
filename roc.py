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

    def CostFunc(self, W, X, y):
        cost = 1/2 * np.average(np.square(self.Hypothesis(W, X) - y))
        regTerm = self.C/2 * np.sum(np.square(W))
        return cost + regTerm
