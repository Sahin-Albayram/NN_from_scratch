import numpy as np


class DenseModel:
    def __init__(self, layer_dims, activation = "relu"):
        self.activation = activation
        self.weights = []
        self.bias = []
        for i in range(len(layer_dims)-1):
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i+1])/ np.sqrt(layer_dims[i-1])) 
            self.bias.append(np.zeros((1, layer_dims[i+1])))

    @staticmethod
    def linear_forward(A, W, b):
        Z = np.dot(A,W) + b
        cache = (A, W, b)
        assert(Z.shape == (W.shape[0], A.shape[1]))
        return Z, cache
    
    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation):
        Z, linear_cache = DenseModel.linear_forward(A_prev, W, b)
        
        if activation == "relu":
            A, activation_cache = DenseModel.relu(Z)
        elif activation == "sigmoid":
            A, activation_cache = DenseModel.sigmoid(Z)
        elif activation == "softmax":
            A, activation_cache = DenseModel.softmax(Z)            
        else:
            raise ValueError("Unsupported activation function")
        
        cache = (linear_cache, activation_cache)
        return A, cache

    def forward(self, X):
        caches = []
        A = X
        for i in range(len(self.weights)):
            if i == len(self.weights) - 1:
                # For the last layer, use softmax activation
                A_last, cache = self.linear_activation_forward(A, self.weights[i], self.bias[i], "softmax")
            else:
                A, cache = self.linear_activation_forward(A, self.weights[i], self.bias[i], self.activation)
            caches.append(cache)
        return A_last, caches
    
    @staticmethod
    def compute_cost(AL, Y):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        #print("cost="+str(cost))
        return cost
    
    def backward(self, d_outputs, learning_rate):
        pass
    
    
    @staticmethod
    def sigmoid(Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A, cache

    @staticmethod
    def relu(Z):
        A = np.maximum(0,Z)    
        cache = Z 
        return A, cache

    @staticmethod
    def softmax(Z):
        e_x = np.exp(Z)
        A= e_x / np.sum(np.exp(Z))  
        cache=Z
        return A,cache  