import numpy as np
import pandas as pd
from tqdm import tqdm

class MultiClassification: 
    
    def one_hot(self, y_value):
        y_new = []
        for i in y_value.values:
            y_n = [0] * self.num_class
            y_n[i] = 1
            y_new.append(y_n)
        y_new = pd.DataFrame(y_new)
        return y_new
    
    def xavier_initialize(self, uniform = True):
        if uniform:
            limit = np.sqrt(6 / (self.num_class + self.features))
            bias = self.bias_initialize()
            return np.random.uniform(-limit, limit, (self.num_class, self.features)), bias
        else:
            std = np.sqrt(2 / (self.num_class + self.features))
            bias = self.bias_initialize()
            return np.random.normal(0, std, (self.num_class, self.features)), bias
    
    def bias_initialize(self):
        return np.zeros((self.num_class, 1))
    
    def random_initialize(self):
        weights = np.random.rand(self.num_class, self.features)
        bias = np.zeros((self.num_class, 1))
        return weights, bias
    
    def initialize(self, init = 'xavier', uniform = True):
        if init == 'xavier':
            w, b = self.xavier_initialize(uniform=uniform)
            return w, b
        return self.random_initialize()
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
    
    def stochastic_gradient(self, xi, yi, activate):
        dW = np.outer((activate - yi), xi)  
        db = activate - yi 
        return dW, db
    
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
        
        np.save('weights.npy', self.weights, allow_pickle=True)
        np.save('bias.npy', self.bias, allow_pickle=True)
        print("Complete Saving Weights and Bias")

    def get_weights_bias(self, init = 'xavier', uniform = True):
        if self.is_file_empty('weights.npy') or self.is_file_empty('bias.npy'):
            print("Generate new weights and bias")
            self.weights, self.bias = self.initialize(init=init, uniform=uniform)
        else:
            print("Loading Weights and Bias")
            self.weights = np.load('weights.npy', allow_pickle=True)
            self.bias = np.load('bias.npy', allow_pickle=True)
        
    def fit(self, x_train, y_train, learning_rate = 0.01, batch_size = 64, epochs = 1000, init = 'xavier', uniform = True, momentum = True):
        self.epochs = epochs
        if type(learning_rate) == int:
            self.learning_rate = learning_rate
        elif type(learning_rate) == list:
            desc_lr_epochs = self.epochs // len(learning_rate)
            self.learning_rate = learning_rate[0]
        self.num_class = len(y_train.unique())
        y_train_one_hot = self.one_hot(y_train)
        
        self.samples, self.features = x_train.shape
        self.get_weights_bias(init=init, uniform=uniform)
        if momentum:
            self.velo_w = 0
            self.velo_b = 0
            
        self.accuracy_point = []
        self.epoch_loss = []
        for epoch in tqdm(range(self.epochs), desc='Epochs '):
            if type(learning_rate) == list and epoch % desc_lr_epochs == 0:
                self.learning_rate = learning_rate[min(epoch // desc_lr_epochs, len(learning_rate) - 1)]
            rand_id = np.random.choice(self.samples, size = batch_size, replace = False)
            for i in rand_id:
                xi = np.array([x_train.iloc[i, :]]).T
                yi = np.array([y_train_one_hot.iloc[i, :]]).T
                linear_combination = np.matmul(self.weights, xi) + self.bias
                activate = self.softmax(linear_combination)
                self.dW, self.db = self.stochastic_gradient(xi, yi, activate)
                self.update_params(momentum = momentum)
            
            if epoch % 100 == 0:
                y_pred = self.predict(x_train)
                acc = self.accuracy(y_train, y_pred)
                self.accuracy_point.append(acc)
                y_pred = self.one_hot(pd.Series(y_pred))
                loss = self.loss(y_train_one_hot.values, y_pred.values)
                self.epoch_loss.append(loss)
                
        self.save_weights_bias()
    
    def momentum(self, alpha = 0.9):
        self.velo_w = alpha*self.velo_w + self.learning_rate*self.dW
        self.velo_b = alpha*self.velo_b + self.learning_rate*self.db
        self.weights = self.weights - self.velo_w
        self.bias = self.bias - self.velo_b
    
    def update_params(self, momentum = True):
        if momentum:
            self.momentum()
        else:
            self.weights = self.weights - self.learning_rate * self.dW
            self.bias = self.bias - self.learning_rate * self.db
    
    def loss(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss_value = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss_value
    
    def predict_single_point(self, xi):
        linear = np.matmul(self.weights, xi) + self.bias
        output = self.softmax(linear)
        digit = np.argmax(output)
        return digit
    
    def predict(self, x):
        y_pred = []
        for i in range(x.shape[0]):
            xi = np.array([x.iloc[i, :]]).T
            linear = np.matmul(self.weights, xi) + self.bias
            output = self.softmax(linear)
            digit = np.argmax(output)
            y_pred.append(digit)
        return np.array(y_pred)
    
    def accuracy(self, y_true, y_pred):
        y_true = y_true.to_numpy()
        c = 0
        for i in range(len(y_pred)):
            if y_pred[i] != y_true[i]:
                c += 1
        return 1-(c/len(y_pred))