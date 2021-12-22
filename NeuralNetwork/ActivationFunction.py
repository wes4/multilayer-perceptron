import abc
import numpy as np

class ActivationFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def activate(self, z):
        pass
    @abc.abstractmethod
    def derivative(self, dLossdA):
        pass

class Sigmoid(ActivationFunction):
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoidDerivative(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))
    def activate(self, z):
        return self.sigmoid(z)
    def derivative(self, z):
        return self.sigmoidDerivative(z)

class Relu(ActivationFunction):
    def relu(self, x):
        return max(0, x)
            # if (x > 0):
            #     return x
            # else:
            #     return 0
    def reluDerivative(self, x):
            if (x < 0):
                return 0
            else:
                return 1
    def activate(self, z):
        vectorizedRelu = np.vectorize(self.relu)
        return vectorizedRelu(z)
    def derivative(self, z):
        vectorizedReluDerivative = np.vectorize(self.reluDerivative)
        dAdZ = vectorizedReluDerivative(z)
        return dAdZ

class Maxout(ActivationFunction):
    def maxout(self, x):
        return max(x, 2*x)
            # if (x > 0):
            #     return x
            # else:
            #     return 0
    def maxoutDerivative(self, x):
            if (x >= 0):
                return 2
            else:
                return 1
    def activate(self, z):
        vectorizedRelu = np.vectorize(self.maxout)
        return vectorizedRelu(z)
    def derivative(self, z):
        vectorizedReluDerivative = np.vectorize(self.maxoutDerivative)
        dAdZ = vectorizedReluDerivative(z)
        return dAdZ
    