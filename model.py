import abc

class Model:
    """ machine learning model
    """

    @abc.abstractmethod
    def Fit(self, X, y):
        """ fit data method """
        return

    @abc.abstractmethod
    def Predict(self, X):
        """ fit data method """
        return

class LogisticRegression(Model):
    """ logistic regression with l2-penalty
    attributes:
        tol: tolerance for stopping criteria
        C: regularization strength
    """
    def __init__(self, tolerance, C):
        self.tol = tolerance
        self.C = C

    def CostFunc(self, X, y):

    def Fit(self, X, y):
