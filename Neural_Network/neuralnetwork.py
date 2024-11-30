import numpy as np
import math
from layer import Layer

class DeepNeuralNetwork:
    
    def __init__(self, *layers):
        self.layers = layers
        self.layers_len = len(layers)
        self.weights = self.initialize()
        self.reshape_bias()
        self.dw, self.db = self.init_derivative()
        
    def initialize(self):
        weight = []
        for i in range(self.layers_len - 1):
            w = (np.random.rand(self.layers[i].get_dim(), self.layers[i + 1].get_dim())) * (math.sqrt(2 / (self.layers[i].get_dim() + self.layers[i + 1].get_dim())))
            weight.append(w)
        return weight
    
    def reshape_bias(self):
        for i in range(self.layers_len - 1):
            if self.layers[i].train_bias:
                self.layers[i].biases = np.zeros((self.weights[i].shape[1], 1))
    
    def init_derivative(self):
        db = []
        dw = []
        for i in range(self.layers_len):
            if self.layers[i].train_bias:
                db.append(np.zeros(self.layers[i].biases.shape))
            if i != self.layers_len - 1:
                dw.append(np.zeros((self.layers[i].get_dim(), self.layers[i + 1].get_dim())))
        return dw, db
    
    def __call__(self, x):
        return self.forward(x)[0][-1]
    
    def forward(self, x):
        activation_cache = [x]
        linear_cache = []
        
        for l in range(1, self.layers_len):
            # print("layer :", l)
            # print("activate :", self.layers[l].activation_func)
            z = np.matmul(self.weights[l-1].T, activation_cache[-1]) + self.layers[l-1].biases
            # print(f"z{l} =", z)
            linear_cache.append(z)
            a = self.layers[l].activate(linear_cache[-1])
            # print(f"a{l+1} =", a)
            activation_cache.append(a)
            
        return activation_cache, linear_cache
    
    def calculate_loss(self, y_true, y_pred):
        return DeepNeuralNetwork.binary_cross_entropy_loss(y_true, y_pred)
    
    def binary_cross_entropy_loss(y_true, y_pred):
        #  Ensure that predicted probabilities are within the valid range (0, 1) to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def error(self, loss, linear_cache):
        errors = [0] * (self.layers_len)
        errors[-1] = np.multiply(self.layers[self.layers_len - 1].derivative(loss), self.layers[self.layers_len - 1].derivative(linear_cache[-1]))
        for i in range(self.layers_len - 2, 0, -1):
            errors[i] = np.multiply(np.matmul(self.weights[i], errors[i + 1]), self.layers[i].derivative(linear_cache[i - 1]))
        return errors[1:]
    
    def backpropagation(self, linear_cache, activation_cache, y_true):
        dZ = activation_cache[-1] - y_true
        for l in range(self.layers_len - 2, -1, -1):
            # print('back prop layer', l)
            # print(f'dZ{l} =\n', dZ)
            self.dw[l] = np.matmul(activation_cache[l], dZ.T)
            # print(f'\ndw{l} =\n', self.dw[l])
            self.db[l] = dZ
            # print(f'\ndb{l} =\n', self.db[l])
            if l > 0:
                dA = np.matmul(self.weights[l], dZ)
                dZ = np.multiply(dA, self.layers[l].derivative(linear_cache[l-1]))
        
    
    def update(self, lr = 0.01):
        for i in range(self.layers_len - 1):
            self.weights[i] = self.weights[i] - lr * self.dw[i]
            self.layers[i].biases = self.layers[i].biases - lr * self.db[i]
    
    def predict(self, x):
        y_pred = []
        for i in x.values:
            i = np.array([i]).T
            output = self.forward(i)[0][-1]
            if output < 0.5:
                y_pred.append(0)
            else:
                y_pred.append(1)
        return y_pred
    
    def accuracy(self, y_true, y_pred):
        y_true = y_true.to_numpy()
        c = 0
        for i in range(len(y_pred)):
            if y_pred[i] != y_true[i]:
                c += 1
        return 1-(c/len(y_pred))