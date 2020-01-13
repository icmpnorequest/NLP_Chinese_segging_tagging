# coding=utf8
"""
@author: Hangyu Xia, Yantong Lai
"""

import sklearn_crfsuite
from sklearn.externals import joblib


class CRF(object):
    def __init__(self, algorithm="lbfgs", p1=0.1, p2=0.1, max_iterations=100):
        self.algorithm = algorithm
        self.p1 = p1
        self.p2 = p2
        self.max_iterations = max_iterations
        self.model = None

    def initialize_model(self):
        """Initialize the model"""
        algorithm = self.algorithm
        p1 = float(self.p1)
        p2 = float(self.p2)
        max_iterations = int(self.max_iterations)
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm, c1=p1, c2=p2,
                                          max_iterations=max_iterations, all_possible_transitions=True)

    def forward(self, features):
        """Input Feature, Output the label"""
        if type(self.model == "<class 'sklearn_crfsuite.estimator.CRF'>"):
            return self.model.predict(features)
        else:
            return None

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


if __name__ == "__main__":
    crf = CRF()         # Default parameters
    crf.initialize_model()
    print(type(crf.model))