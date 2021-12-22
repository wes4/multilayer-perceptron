import abc

class OptimizationAlgorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def optimize(self, Xb, dLossYhat, outputLayer, layers):
        pass

class Backpropagation(OptimizationAlgorithm):
    def optimize(self, Xb, dLossdYhat, outputLayer, layers):
        i = len(layers) - 1

        # Pass through output layer
        outputLayer.Backward(dLossdYhat, layers[len(layers) - 1].Ab)
        
        # For each hidden layer, going backwards
        while (i >= 0):
            if (i == len(layers) - 1): # The last hidden layer before the output layer
                if (i - 1 < 0): # There is only a single hidden layer (A of the previous layer is the inputs)
                    layers[i].Backward(dLossdYhat, outputLayer.W, Xb)
                else: # The last hidden layer before the output layer
                    layers[i].Backward(dLossdYhat, outputLayer.W, layers[i-1].Ab)
            elif (i == 0): # The first hidden layer
                layers[i].Backward(layers[i+1].dLossdZ, layers[i+1].W, Xb)
            else: # All other hidden layers
                layers[i].Backward(layers[i+1].dLossdZ, layers[i+1].W, layers[i-1].Ab)
            i = i - 1

        # Update the weights and biases of the hidden layers and output layer
        outputLayer.UpdateWeightsAndBiases()    
        for layer in layers:
            layer.UpdateWeightsAndBiases()
        
