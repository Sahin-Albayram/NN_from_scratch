import numpy as np
from types import List

class MlpScratch:
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 layer_sizes: List[int]) -> None:
        self.input_size = input_size
        self.otput_size = output_size
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.param_init_std = 0.02
    
    def _create_model(self):
        self.layers = []
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(self._create_layer(self.input_size, self.layer_sizes[i]))
            else:
                self.layers.append(self._create_layer(self.layer_sizes[i-1], self.layer_sizes[i]))
    
    def _create_layer(self,input_size: int, output_size: int):
        weights = np.random.randn(0,self.param_init_std,(output_size, input_size))
        biases = np.random.normal(0,self.param_init_std,(1,output_size))
        return weights, biases 
    
    def _initialize_weights(self):
        for i in range(self.num_layers):
            self.layers[i][0] = np.random.normal(0,0.2,self.layers[i][0].shape[0], self.layers[i][0].shape[1])
            self.layers[i][1] = np.random.randn(0,0.2,self.layers[i][1].shape[0])
    
    def forward(self):
        
    
    def backward_propogation(self):
        pass
    
    