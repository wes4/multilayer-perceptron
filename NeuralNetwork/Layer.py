import numpy as np
from NeuralNetwork.Enumerations import ActivationFunctionType
from NeuralNetwork.Enumerations import WeightInitializationType
from NeuralNetwork.ActivationFunction import ActivationFunction
from NeuralNetwork.ActivationFunction import Sigmoid
from NeuralNetwork.ActivationFunction import Relu
from NeuralNetwork.ActivationFunction import Maxout

class Layer():
    def __init__(self, activationFunction, neuronsInLayer, neuronsInPreviousLayer, weightInitialization, learningRate):
        # Determine Activation Function based on constructor parameter
        if (activationFunction == ActivationFunctionType.Relu):
            self.activationFunction = Relu()
        elif (activationFunction == ActivationFunctionType.Maxout):
            self.activationFunction = Maxout()
        else:
            self.activationFunction = Sigmoid()
        
        # Learning Rate from 10^-3 to 10^-5
        self.learningRate = learningRate

        # Populate weights and biases
        self.Wb = self.GenerateWeightsAndBias(weightInitialization, neuronsInPreviousLayer + 1, neuronsInLayer) # + 1 to account for bias added to the previous layer neurons
        self.W = self.Wb[0:len(self.Wb)-1] # weight matrix without biases to be used in back pass

    # previousLayerAb (batch size, number of neurons in previous layer + 1) = the previous hidden layer's A matrix appended with the bias (or the inputs appended with the bias if this is the first layer)
    def Forward(self, previousLayerAb):
        # Multiply previous layer neurons by the weight matrix (including biases)
        # Z (batch size, neurons in current layer) = previousLayerAb (batch size, neurons in previous layer + 1) dot Wb (neurons in previous layer + 1, neurons in current layer)
        self.Z = np.matmul(previousLayerAb, self.Wb)
        # Apply activation function
        self.A = self.activationFunction.activate(self.Z)
        # Append bias neuron
        self.Ab = np.append(self.A, np.ones((len(self.A), 1)), 1)

    # nextLayerdLossdZ (batch size, number of neurons in next layer) = the derivative of the loss with respect to the next layer's Z
    # nextLayerW (number of neurons in current layer, number of neurons in next layer) = the next layer's weight matrix (not including the biases) (this is dLoss/dY^ if this is the last hidden layer)
    # previousLayerAb (batch size, number of neurons in previous layer + 1) = the previous hidden layer's A matrix appended with the bias (or the inputs appended with the bias if this is the first layer)
    def Backward(self, nextLayerdLossdZ, nextLayerW, previousLayerAb):
        # dLoss/dA (batch size, neurons in current layer) = nextLayerdLossdZ (batch size, neurons in next layer) * nextLayerW.transpose() (neurons in next layer, neurons in current layer)
        self.dLossdA = np.matmul(nextLayerdLossdZ, nextLayerW.transpose())
        self.dAdZ = self.activationFunction.derivative(self.Z)
        self.dLossdZ = np.multiply(self.dLossdA, self.dAdZ)
        # dLoss/dWb (previousLayerNodeCount + 1, currentLayerNodeCount) = previousLayerAb.transpose (neurons in previous layer + 1, batch size) * dLossdZ (batch size, neurons in current layer)
        self.dLossdWb = np.matmul(previousLayerAb.transpose(), self.dLossdZ)
        self.dLossdW = self.dLossdWb[0:len(self.dLossdWb)-1]
        
    def GenerateWeightsAndBias(self, weightInitialization, rows, columns):
        # populate weights and biases with zero or gaussian normal based on constructor parameter
        if (weightInitialization == WeightInitializationType.Zeros):
            return np.zeros(shape=(rows, columns))
        else:
            return np.random.normal(0, 1, (rows, columns))

    # Update the current Wb (weights + biases) based on the calculated dLoss/dWb
    def UpdateWeightsAndBiases(self):
        # Wb(new) = Wb(old) - (learning rate)*(dLoss/dWb) (neurons in previous layer + 1, neurons in current layer)
        newWeights = np.subtract(self.Wb, self.learningRate * self.dLossdWb)
        self.Wb = newWeights
        self.W = self.Wb[0:len(self.Wb)-1]