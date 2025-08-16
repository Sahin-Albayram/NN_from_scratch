import numpy as np


class DenseModel:
    def __init__(self, layer_dims):
        self.weights = []
        self.bias = []
        for i in range(len(layer_dims)-1):
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i+1]))
            self.bias.append(np.zeros((1, layer_dims[i+1])))

    def forward(self, inputs):
        self.inputs = inputs
        for i in range(len(self.weights)):
            inputs = np.dot(inputs, self.weights[i]) + self.bias[i]
        return inputs

    def backward(self, d_outputs, learning_rate):
        pass