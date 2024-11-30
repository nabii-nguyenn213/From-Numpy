import numpy as np

class SGD:
    
    def __init__(self, model : object, learning_rate = 0.01):
        self.model = model
        self.learning_rate = learning_rate
    
    def step(self):
        for i in range(self.model.get_layer_length()):
            if self.model._layers[i].weights is not None:
                self.model._layers[i].weights = self.model._layers[i].weights - self.learning_rate * self.model.backward_caches[i]['dW']
            if hasattr(self.model._layers[i], 'bias'):
                self.model._layers[i].bias = self.model._layers[i].bias - self.learning_rate * self.model.backward_caches[i]['db']