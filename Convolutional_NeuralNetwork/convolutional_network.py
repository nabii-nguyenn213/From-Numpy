from layers import *
from optimizer import SGD

class Convolutional_Neural_Network:
    
    def __init__(self, *layers):
        self._layers = list(layers) if layers else []
        
    def add(self, layer):
        '''
        adding layers to CNN
        '''
        self._layers.append(layer)
    
    def get_layer_length(self):
        return len(self._layers)
    
    def forward_pass(self, input_img):
        if self._layers == []:
            return
        self.forward_caches = []
        '''
            forward caches contain every forward cache of each layer :
                # ? Convolutional layer cache contains  : weights     , output (trainable parameters : weights)
                # ? Maxpooling layer cache contains     : max_indicies, output (non-trainable parameters)
                # ? Averagepooling layer cache contains :               output (non-trainable parameters)
                # ? Flatten layer cache contains        : input shape , output (non-trainable parameters)
                # ? Dense layer cache contains          : linear      , output (trainable parameters : weights, biases)
        '''
        self.forward_caches.append({'output' : input_img})
        
        output = self._layers[0].forward(input_img)
        self.forward_caches.append(self._layers[0].forward_cache)
        
        for layer in range(1, self.get_layer_length()):
            output = self._layers[layer].forward(output, input_shape=output.shape)
            self.forward_caches.append(self._layers[layer].forward_cache)

        return output
    
    def backpropagation(self, y_true):
        self.backward_caches = [None] * self.get_layer_length()
        '''
            backward caches contain every backward cache of each layer :
                # ? Convolutional layer cache contains  : dW          , dA (trainable parameters : weights)
                # ? Maxpooling layer cache contains     :             , dA (non-trainable parameters)
                # ? Averagepooling layer cache contains :             , dA (non-trainable parameters)
                # ? Flatten layer cache contains        :             , dA (non-trainable parameters)
                # ? Dense layer cache contains          : dW   , db   , dA (trainable parameters : weights, biases)
        '''
        y_pred = self.forward_caches[-1]['output']
        
        # ! index layer     =    0       1     2     3      4     5      6      7      8
        # ! forward_caches  = [input, convo, avg, convo, avg, convo, flatten, dense, dense]  : forward caches length = 9
        # ! backward_caches = [convo, avg, convo, avg, convo, flatten, dense, dense]         : backward caches length = 8
        # ! layers          = [convo, avg, convo, avg, convo, flatten, dense, dense]         : layers length = 8
        
        for i in range(self.get_layer_length() -1, -1, -1):
            # * i iterate from 7 -> 0
            if i == self.get_layer_length() - 1: 
                # ? output layer
                self._layers[i].backprop(previous_layer_cache=self.forward_caches[i], next_layer_cache=None, weights_next_layer=None, output_layer=y_pred-y_true)
                self.backward_caches[i] = self._layers[i].backward_cache
            else:
                # ? hidden layers
                self._layers[i].backprop(previous_layer_cache=self.forward_caches[i], next_layer_cache=self.backward_caches[i + 1], weights_next_layer=self._layers[i + 1].weights, output_layer=None)
                self.backward_caches[i] = self._layers[i].backward_cache
    
    def predict(self, input_img):
        output = self._layers[0].forward(input_img, input_shape=-1, predict=True)
        
        for layer in range(1, self.get_layer_length()):
            output = self._layers[layer].forward(output, input_shape=output.shape, predict=True)
        return output
    
    def accuracy(self, y_true, y_pred):
        w = 0
        for i in range(len(y_pred)):
            if np.argmax(y_pred[i]) != np.argmax(y_true[i]):
                w += 1
        return 1 - w/len(y_pred)
    
    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss_value = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss_value