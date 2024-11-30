import numpy as np
import math
from Layers import Layer
from tqdm import tqdm
import pandas as pd

class DeepNeuralNetwork:
    
    def __init__(self, *layers, init = 'xavier', uniform = True):
        self.layers = layers
        self.layers_len = len(layers)
        self.empty_file() # !uncomment this to retraining
        self.get_weights_bias(init=init, uniform=uniform)
        self.dw, self.db = self.init_derivative()
    
    def xavier_initialize(self, uniform = True):
        weights = []
        
        if uniform:
            for i in range(self.layers_len - 1):
                limit = np.sqrt(6 / (self.layers[i].get_dim() + self.layers[i + 1].get_dim()))
                w = np.random.uniform(-limit, limit, (self.layers[i].get_dim(), self.layers[i + 1].get_dim()))
                weights.append(w)
        else:
            for i in range(self.layers_len - 1):
                std = np.sqrt(2 / (self.layers[i].get_dim() + self.layers[i + 1].get_dim()))
                w = np.random.normal(0, std, (self.layers[i].get_dim(), self.layers[i + 1].get_dim()))
                weights.append(w)
                
        return weights
    
    def initialize(self, init, uniform):
        if init == 'xavier':
            weight = self.xavier_initialize(uniform=uniform)
        else:
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
            z = np.matmul(self.weights[l-1].T, activation_cache[-1]) + self.layers[l-1].biases
            # print(f'z{l-1} shape :', z.shape)
            linear_cache.append(z)
            a = self.layers[l].activate(linear_cache[-1])
            # print(f'a{l} shape :', a.shape)
            activation_cache.append(a)
            
        return activation_cache, linear_cache
    
    def backpropagation(self, linear_cache, activation_cache, y_true):
        dZ = activation_cache[-1] - y_true
        for l in range(self.layers_len - 2, -1, -1):
            self.dw[l] = np.matmul(activation_cache[l], dZ.T)
            self.db[l] = dZ
            if l > 0:
                dA = np.matmul(self.weights[l], dZ)
                dZ = np.multiply(dA, self.layers[l].derivative(linear_cache[l-1]))
    
    def update(self):
        for i in range(self.layers_len - 1):
            self.weights[i] = self.weights[i] - self.learning_rate * self.dw[i]
            self.layers[i].biases = self.layers[i].biases - self.learning_rate * self.db[i]
    
    def predict(self, x):
        y_pred = []
        for i in x.values:
            i = np.array([i]).T
            output = self.forward(i)[0][-1]
            digit = np.argmax(output)
            y_pred.append(digit)
        return np.array(y_pred)
    
    def predict_single_point(self, x):
        output = self.forward(x)[0][-1]
        digit = np.argmax(output)
        return digit
    
    def accuracy(self, y_true, y_pred):
        y_true = y_true.to_numpy()
        c = 0
        for i in range(len(y_pred)):
            if y_pred[i] != y_true[i]:
                c += 1
        return 1-(c/len(y_pred))
    
    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss_value = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss_value
    
    def one_hot(self, y_value):
        num_class = len(y_value.unique())
        y_new = []
        for i in y_value.values:
            y_n = [0] * num_class
            y_n[i] = 1
            y_new.append(y_n)
        y_new = pd.DataFrame(y_new)
        return y_new
    
    def fit(self, x_train, y_train, lr = 0.01, batch_size = 32, epochs = 1000):
        self.epochs = epochs
        if isinstance(lr, float):
            self.learning_rate = lr
        elif isinstance(lr, list):
            desc_lr_epochs = self.epochs // len(lr)
            self.learning_rate = lr[0]
        N, d = x_train.shape
        self.accuracy_point = []
        y_train_one_hot = self.one_hot(y_train)
        for epoch in tqdm(range(self.epochs), desc='Epochs', ascii=True):
            if isinstance(lr, list) and epoch % desc_lr_epochs == 0:
                self.learning_rate = lr[min(epoch // desc_lr_epochs, len(lr) - 1)]
            rand_id = np.random.choice(N, size=batch_size, replace=False)
            for i in rand_id:
                xi = np.array([x_train.iloc[i, :]]).T
                yi = np.array([y_train_one_hot.iloc[i, :]]).T
                activation_cache, linear_cache = self.forward(xi)
                self.backpropagation(linear_cache, activation_cache, yi)
                self.update()
            if epoch % 100 == 0:
                y_pred = self.predict(x_train)
                self.acc = self.accuracy(y_train, y_pred)
                self.accuracy_point.append(self.acc)

        self.save_weights_bias()
    
    def empty_file(self):
        empty_array = np.array([])
        np.save('weights.npy', empty_array)
        np.save('bias.npy', empty_array)
    
    def is_file_empty(self, file_path):
        try:
            data = np.load(file_path, allow_pickle=True)
            return data.size == 0
        except ValueError:
            print(f"Invalid or corrupted .npy file: {file_path}")
            return True
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return True

        
    def save_weights_bias(self):
        self.empty_file()
        
        bias = []
        for i in range(self.layers_len-1):
            bias.append(self.layers[i].biases)
        
        np.save('weights.npy', np.array(self.weights, dtype=object), allow_pickle=True)
        np.save('bias.npy', np.array(bias, dtype=object), allow_pickle=True)
        print("Complete Saving Weights and Bias")

    def get_weights_bias(self, init = 'xavier', uniform = True):
        if self.is_file_empty('weights.npy') or self.is_file_empty('bias.npy'):
            print("Generate new weights and bias")
            self.weights = self.initialize(init=init, uniform=uniform)
            self.reshape_bias()
        else:
            print("Loading Weights and Bias")
            self.weights = list(np.load('weights.npy', allow_pickle=True))
            bias = list(np.load('bias.npy', allow_pickle=True))
            for i in range(self.layers_len - 1):
                self.layers[i].biases = bias[i]