import abc
import numpy as np

class LossFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculateLoss(self, calculatedOutput, expectedOutput):
        pass

class Softmax(LossFunction):
    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x/np.sum(exp_x)

    def calculateLoss(self, calculatedOutputBatch, expectedOutputBatch):
        lossSumPerBatch = 0
        successes = 0
        
        # Create dLoss/dY^ with correct size
        dLossdYhat = np.copy(calculatedOutputBatch)

        # for each input/output pair
        for i in range(len(calculatedOutputBatch)):
            # Get the input/output pair
            calculatedUnnormalizedOutput = calculatedOutputBatch[i]
            expectedNormalizedOutput = expectedOutputBatch[i]

            # Apply softmax to unnormalized outputs
            calculatedNormalizedOutput = self.softmax(calculatedUnnormalizedOutput)


            # Determine if calculated output is correct (index of 1 in expected output = index of highest value in calculated normalized output)
            correctIndex = np.argmax(expectedNormalizedOutput)
            if(np.argmax(calculatedNormalizedOutput) == correctIndex):
                successes = successes + 1
            
            # Calculate Softmax Loss using Cross Entropy = (-1 * Y[i] * ln(Y^[i]))
            # Since Y[i] = 0 for all indexes except the classification index j (Y[j] = 1), the resulting vector is 0 except for the j index (which is -ln(Y^[j])) which can be calculated using max()
            loss = sum(-1 * (expectedNormalizedOutput * np.log(calculatedNormalizedOutput)))

            # Calculate Loss with respect to Y^
            dLossdYhat[i] = calculatedNormalizedOutput - expectedNormalizedOutput

            # Sum the losses across the batch
            lossSumPerBatch = lossSumPerBatch + loss

        return lossSumPerBatch, successes, dLossdYhat/len(calculatedOutputBatch)

class HingeLoss(LossFunction):
    def calculateLoss(self, calculatedOutputBatch, expectedOutputBatch):
        lossSumPerBatch = 0
        successes = 0
        
        # Create dLoss/dY^ with correct size
        dLossdYhat = np.copy(calculatedOutputBatch)

        # for each input/output pair
        for i in range(len(calculatedOutputBatch)):
            # Get the input/output pair
            calculatedOutput = calculatedOutputBatch[i]
            expectedOutput = expectedOutputBatch[i]
            dLossdYhatRow = dLossdYhat[i]

            # Determine if calculated output is correct (index of 1 in expected output = index of highest value in calculated normalized output)
            correctIndex = np.argmax(expectedOutput)
            if(np.argmax(calculatedOutput) == correctIndex):
                successes = successes + 1
            
            # Calculate Hinge Loss Σ max(0, Y^[j](j != Sj) - Y^[Sj] + 1)
            # Calculate Loss with respect to Y^ 
            loss = 0
            indexesThatAffectLoss = 0
            for j in range(len(calculatedOutput)):
                if (j != correctIndex):
                    value = max(0, calculatedOutput[j] - calculatedOutput[correctIndex] + 1)
                    loss = loss + value
                    if (value <= 0): # If max <= 0, dLoss/dY^ index is 1 because the value doesn't affect the loss
                        dLossdYhatRow[j] = 0
                    else: # If max is 0, dLoss/dY^ index is 0 because the value 
                        dLossdYhatRow[j] = 1
                        indexesThatAffectLoss = indexesThatAffectLoss + 1 # keep track of how many indexes affect loss to determine the loss at the correct index
            dLossdYhatRow[correctIndex] = -1 * indexesThatAffectLoss

            # Sum the losses across the batch
            lossSumPerBatch = lossSumPerBatch + loss

        return lossSumPerBatch, successes, dLossdYhat/len(calculatedOutputBatch)