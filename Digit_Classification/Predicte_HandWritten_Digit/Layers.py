import numpy as np

class Layer:
    
    def __init__(self, dim, activation_func = "linear", train_bias = True):
        self.dim = dim
        self.activation_func = activation_func
        self.train_bias = train_bias
        if train_bias:
            self.biases = np.zeros(dim)
    
    def activate(self, x):
        return Layer.activation_function(x, self.activation_func)
    
    def derivative(self, x):
        return Layer.derivative_activation_function(x, self.activation_func)
    
    def activation_function(x, function):
        if function == "relu":
            return np.maximum(0, x)
        elif function == "sigmoid":
            return 1/(1 + np.exp(-x))
        elif function == "tanh":
            return np.tanh(x)
        elif function == 'softmax':
            sum = np.exp(x).sum()
            return np.exp(x) / sum
        else:
            return x
        
    def derivative_activation_function(x, function):
        if function == "relu":
            return np.where(x < 0, 0, 1)
        elif function == "sigmoid":
            return Layer.activation_function(x, function) * (1 - Layer.activation_function(x, function))
        elif function == "tanh":
            return 1 - np.tanh(x) ** 2
        elif function == 'softmax':
            sm = Layer.activate_func(x, function)
            if x.ndim == 1:
                dx = np.einsum('i,j->ij', -sm, sm)
                for i in range(0, len(x)):
                    dx[i][i] += sm[i]
            else:
                dx = np.einsum('ij,ik->ijk', -sm, sm)
                for i in range(0, x.shape[0]):
                    for j in range(0, x.shape[1]):
                        dx[i][j][j] += sm[i][j]
            return dx
        else:
            return np.ones(x.shape)
    
    def get_dim(self):
        return self.dim
