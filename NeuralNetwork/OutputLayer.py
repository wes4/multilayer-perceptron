import numpy as np
from NeuralNetwork.Enumerations import LossFunctionType
from NeuralNetwork.Enumerations import WeightInitializationType
from NeuralNetwork.LossFunction import LossFunction
from NeuralNetwork.LossFunction import Softmax
from NeuralNetwork.LossFunction import HingeLoss

class OutputLayer():
    def __init__(self, lossFunctionType, neuronsInLayer, neuronsInPreviousLayer, weightInitialization, learningRate):
        # Learning Rate from 10^-3 to 10^-5
        self.learningRate = learningRate
        
        # Determine loss function based on constructor parameter
        if (lossFunctionType == LossFunctionType.Softmax):
            self.lossFunction = Softmax()
        else:
            self.lossFunction = HingeLoss()

        # Populate weights and biases
        self.Wb = self.GenerateWeightsAndBiases(weightInitialization, neuronsInPreviousLayer + 1, neuronsInLayer) # + 1 to account for bias in previous layer as a neuron
        self.W = self.Wb[0:len(self.Wb)-1] # weight matrix without biases to be used in back pass

    # previousLayerAb (batch size, number of neurons in previous layer + 1) = the previous hidden layer's A matrix appended with the bias (or the inputs appended with the bias if this is the first layer)
    def Forward(self, previousLayerAb):
        # Yhat (batchSize, neurons in output layer) = previousLayerAb (batchSize, neurons in previous layer + 1) dot Wb (neurons in previous layer + 1, neurons in output layer)
        self.Yhat = np.matmul(previousLayerAb, self.Wb)

    # dLossdYhat (batch size, number of neurons in the output layer) = the derivative of the loss with respect to Y^
    # previousLayerAb (batch size, number of neurons in previous layer + 1) = the previous hidden layer's A matrix appended with the bias (or the inputs appended with the bias if this is the first layer)
    def Backward(self, dLossdYhat, previousLayerAb):
        # dLossdWb (neurons in previous layer + 1, neurons in output layer) = previousLayerAb.transpose (neurons in previous layer + 1, batchSize) * dLossdYhat (batchSize, neurons in output layer)
        self.dLossdWb = np.matmul(previousLayerAb.transpose(), dLossdYhat)
        self.dLossdW = self.dLossdWb[0:len(self.dLossdWb)-1]
        
    def GenerateWeightsAndBiases(self, weightInitialization, rows, columns):
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