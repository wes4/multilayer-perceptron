import numpy
import math
from NeuralNetwork.Layer import Layer
from NeuralNetwork.OutputLayer import OutputLayer
from NeuralNetwork.Enumerations import LossFunctionType
from NeuralNetwork.LossFunction import Softmax
from NeuralNetwork.LossFunction import HingeLoss
from NeuralNetwork.OptimizationAlgorithm import Backpropagation

class NeuralNetwork():
    def __init__(self, numberOfInputs, numberOfOutputs, numberOfNodesPerLayer, activationFunctionTypePerLayer, weightInitializationType, lossFunctionType, learningRate=None):
        self.inputCount = numberOfInputs
        self.outputCount = numberOfOutputs
        self.hiddenLayerCount = len(numberOfNodesPerLayer)
        self.nodesPerLayer = numberOfNodesPerLayer
        self.weightInitializationType = weightInitializationType
        self.activationFunctionTypePerLayer = activationFunctionTypePerLayer
        self.lossFunctionType = lossFunctionType
        self.layers = []

        # Default Optimization Algorithm
        self.optimizationAlgorithm = Backpropagation()

        # Default Learning Rate
        if(learningRate == None):
            self.learningRate = 0.001
        else:
            self.learningRate = learningRate

        # Determine Loss Function
        if (self.lossFunctionType == LossFunctionType.Softmax):
            self.lossFunction = Softmax()
        else:
            self.lossFunction = HingeLoss()

        # Create Hidden Layers
        for i in range(self.hiddenLayerCount):
            if (i == 0): # First hidden layer takes input
                self.layers.append(Layer(self.activationFunctionTypePerLayer[i], self.nodesPerLayer[i], self.inputCount, self.weightInitializationType, self.learningRate))
            else: # All other layers take output from the previous layer
                self.layers.append(Layer(self.activationFunctionTypePerLayer[i], self.nodesPerLayer[i], self.nodesPerLayer[i-1], self.weightInitializationType, self.learningRate))

        # Create Output Layer
        self.outputLayer = OutputLayer(self.lossFunctionType, self.outputCount, self.nodesPerLayer[len(self.layers)-1], self.weightInitializationType, self.learningRate)

    # Given a set of inputs, expected outputs, batch size, and number of epochs, 
    # divide the inputs and expected outputs into batches based on the batch size
    # and pass all the batches through the neural network, performing gradient descent once per batch
    # Repeat for each epoch
    # Average Loss and Error Rate per epoch are tracked
    # Predict validation inputs given outputs for each epoch to calculate accuracy for weights at that epoch
    def Train(self, trainingX, trainingY, batchSize, numberOfEpochs, validationX=None, validationY=None):
        self.batchSize = batchSize
        self.trainingLossPerEpoch = []
        self.trainingErrorRatePerEpoch = []
        self.validationLossPerEpoch = []
        self.validationErrorRatePerEpoch = []
        self.weightsAtEpoch = []

        # For each epoch
        for i in range(numberOfEpochs):
            totalLossSum = 0
            totalSuccesses = 0

            # First evaluate current weights using validation data
            if (validationX is not None and validationY is not None):
                validationLoss, validationErrorRate, _ = self.Predict(validationX, validationY)
                self.validationLossPerEpoch.append(validationLoss)
                self.validationErrorRatePerEpoch.append(validationErrorRate)

            # Divide input and expected output into batches based on batch size
            inputBatches = numpy.array_split(trainingX, math.ceil(len(trainingX) / self.batchSize))
            expectedOutputBatches = numpy.array_split(trainingY, math.ceil(len(trainingY) / self.batchSize))

            # Save weights and biases being used for the current epoch
            currentWeights = []
            for i in range(len(self.layers)):
                currentWeights.append(numpy.copy(self.layers[i].Wb))
            currentWeights.append(numpy.copy(self.outputLayer.Wb))
            self.weightsAtEpoch.append(currentWeights)

            # For each batch
            for j in range(len(inputBatches)):
                # Get the current batch
                currentBatch = inputBatches[j]

                # Append biases
                Xb = numpy.append(currentBatch, numpy.ones((len(currentBatch), 1)), 1)

                # Execute Forward Pass
                self.ForwardPass(Xb)

                # Calculate Average Loss and Accuracy per batch
                sumOfLossPerBatch, batchSuccesses, self.dLossdYhat = self.lossFunction.calculateLoss(self.outputLayer.Yhat, expectedOutputBatches[j])

                totalSuccesses = totalSuccesses + batchSuccesses
                totalLossSum = totalLossSum + sumOfLossPerBatch

                # Perform Optimization (Backpropagation)
                self.optimizationAlgorithm.optimize(Xb, self.dLossdYhat, self.outputLayer, self.layers)
            
            # Average the total loss sum / number of inputs in an epoch
            self.trainingLossPerEpoch.append(totalLossSum / len(trainingX))

            # Compute the total number of successes / total number of inputs
            self.trainingErrorRatePerEpoch.append(1 - (totalSuccesses / len(trainingX)))

    # Perform the forward pass, where Xb is the input batch appended with a 1
    # Not intended to be directly called outside the API
    def ForwardPass(self, Xb):
        # Pass through all the hidden layers
        for i in range(self.hiddenLayerCount):
            if (i == 0): # The first hidden layer
                self.layers[i].Forward(Xb)
            else: # All other hidden layers
                self.layers[i].Forward(self.layers[i-1].Ab)

        # Pass to the output layer
        self.outputLayer.Forward(self.layers[len(self.layers)-1].Ab)

    # Allow a user to specify the weights used at a specific epoch, where epochOfWeightsToUse is an integer corresponding to an epoch number
    # To be used before Predict() with validation or test data to verify loss and error rate of a specific epoch 
    def SetWeightsFromEpoch(self, epochOfWeightsToUse):
        # Verify the specified epoch is valid
        if (len(self.weightsAtEpoch) > epochOfWeightsToUse):
            weights = self.weightsAtEpoch[epochOfWeightsToUse]

            for j in range(len(weights) - 2): # The last weight matrix is for the output layer
                self.layers[j].Wb = weights[j] # Update the hidden layer weights

            # Update the output layer weights
            self.outputLayer.W = weights[len(weights)-1]

    # Allow a user to predict the outputs given a set of inputs
    def Predict(self, X, Y):
        Xb = numpy.append(X, numpy.ones((len(X), 1)), 1)

        self.ForwardPass(Xb)

        sumOfLoss, successes, _ = self.lossFunction.calculateLoss(self.outputLayer.Yhat, Y)

        return sumOfLoss / len(X), (1 - (successes / len(X))), self.outputLayer.Yhat

    def GetTrainingLossPerEpoch(self):
        return self.trainingLossPerEpoch
    
    def GetTrainingErrorRatePerEpoch(self):
        return self.trainingErrorRatePerEpoch

    def GetValidationLossPerEpoch(self):
        return self.validationLossPerEpoch

    def GetValidationErrorRatePerEpoch(self):
        return self.validationErrorRatePerEpoch