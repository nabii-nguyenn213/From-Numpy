from layers import *

class Sequential:
    
    def __init__(self, *layers, regularization = None):
        self._layers = list(layers) if layers else []
        self.regularization = self._valid_regulation(regularization=regularization)
        
    def add(self, layer):
        '''
        adding layers to CNN
        '''
        self._layers.append(layer)
    
    def _valid_regulation(self, regularization):
        if regularization in ['l1', 'l2', None]:
            return regularization
        else :
            raise ValueError(f"Support l1, l2 regularization, got {regularization}")

    def get_layer_length(self):
        return len(self._layers)

    def parameters(self):
        weights = []
        for i in range(self.get_layer_length()):
            if self._layers[i].weights is not None:
                weights.append(self._layers[i].weights)
            else:
                weights.append(0)
        return weights
    
    def forward_pass(self, input_img):
        if self._layers == []:
            return
        self.forward_caches = []

        self.forward_caches.append({'output' : input_img})
        
        output = self._layers[0].forward(input_img)
        # print("Forward pass: ")
        # print("output layer 1 shape :", output)
        self.forward_caches.append(self._layers[0].forward_cache)
        for layer in range(1, self.get_layer_length()):
            output = self._layers[layer].forward(output, input_shape=output.shape)
            # print(f'output layer {layer + 1} shape :', output)
            self.forward_caches.append(self._layers[layer].forward_cache)
        # print("Done forward pass")
        return output
    
    def backpropagation(self, y_true, regularization_term=0):
        self.backward_caches = [None] * self.get_layer_length()
        y_pred = self.forward_caches[-1]['output']
        # print("Backward pass: ") 
        for i in range(self.get_layer_length() -1, -1, -1):
            # print("layer :", i+1)
            # * i iterate from 7 -> 0
            if i == self.get_layer_length() - 1: 
                # ? output layer
                self._layers[i].backprop(previous_layer_cache=self.forward_caches[i], next_layer_cache=None, weights_next_layer=None, output_layer=y_pred-y_true)
                self.backward_caches[i] = self._layers[i].backward_cache
            else:
                # ? hidden layers
                self._layers[i].backprop(previous_layer_cache=self.forward_caches[i], next_layer_cache=self.backward_caches[i + 1], weights_next_layer=self._layers[i + 1].weights, output_layer=None)
                self.backward_caches[i] = self._layers[i].backward_cache
        # print("Done backward pass")
        if self.regularization == 'l1' or self.regularization == 'l2':
            for i in range(self.get_layer_length()):
                if self._layers[i].weights is not None:
                    self.backward_caches[i]['dW'] += regularization_term[i]
                    
    def predict(self, input_img):
        output = self._layers[0].forward(input_img, input_shape=-1, predict=True)
        
        for layer in range(1, self.get_layer_length()):
            output = self._layers[layer].forward(output, input_shape=output.shape, predict=True)
        return output
    
