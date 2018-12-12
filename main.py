import numpy as np
import sys

class PartyNN(object):
    def __init(self, learnRate = 0.1):
        self.axonsInput = np.random.normal(0.0, 1, (2, 3))
        self.axonsHidden = np.random.normal(0.0, 1, (1, 2))
        self.sigmoidMapper = np.vectorize(self.sigmoid)
        self.learningRate = np.array([learnRate])

    def sigmoid(self, x):
        return 1 / ( 1 + np.exp(-x) )

    def predict(self, inputs, axons, last = False):
        in_data = np.dot(axons, inputs)
        out_data = self.sigmoidMapper(in_data)
        return out_data if last else self.predict(out_data, self.axonsHidden, True)

    def train(self, inputs):

