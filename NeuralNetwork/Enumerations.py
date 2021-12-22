import enum

class ActivationFunctionType(enum.Enum):
    Sigmoid = 0
    Relu = 1
    Maxout = 2

class WeightInitializationType(enum.Enum):
    Zeros = 0
    Gaussian = 1

class LossFunctionType(enum.Enum):
    Softmax = 0
    HingeLoss = 1

class OptimizationAlgorithmType(enum.Enum):
    Backpropagation = 0