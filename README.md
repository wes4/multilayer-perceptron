# multilayer-perceptron
A Multilayer Perception created from scratch (essentially numpy only) for UAH CS637 Deep Learning.

NeuralNetwork
- NeuralNetwork.py - provides an interface for creating a NeuralNetwork object, which is a multilayer perceptron.

- Layer.py - defines a layer in a multilayer perceptron. This can be a hidden layer or the input layer.

- OutputLayer.py - defines the output layer in a multilayer perceptron.

- LossFunction.py - abstracts away the loss function, so that it can be injected when creating the NeuralNetwork object. Currently implemented options are Softmax and HingeLoss.

- OptimizationAlgorithm.py - abstracts away the optimization algorithm, so that it can be injected when creating the NeuralNetwork object. The only option currently implemented is Backpropagation.

- ActivationFunction.py - abstracts away the activation function, so that it can be injected on a by layer basis when creating the NeuralNetwork object. Currently implemented options are Sigmoid, Relu, and Maxout.
