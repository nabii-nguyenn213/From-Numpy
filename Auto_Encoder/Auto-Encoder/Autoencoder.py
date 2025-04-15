from Convolutional_NeuralNetwork import layers
import numpy as np
from Convolutional_NeuralNetwork.layers import Layers
from Convolutional_NeuralNetwork.convolutional_network import Convolutional_Neural_Network

class Encoder:
    def __init__(self, latent_dim = 16):
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()

    def build_encoder(self):
        encoder = Convolutional_Neural_Network()
        encoder.add(Layers.Convo(num_filter=16, kernel_size=(3, 3), activation='relu', stride=2, padding='same', input_shape=(1, 28, 28, 1)))
        encoder.add(Layers.Convo(num_filter=32, kernel_size=(3, 3), activation='relu', stride=2, padding='same', input_shape=(16, 28, 28, 1)))
        encoder.add(Layers.Flatten())
        encoder.add(Layers.Dense(dim=(23328, self.latent_dim), activation='linear', train_bias=True, xavier_uniform=True))
        return encoder

    def forward(self, x):
        return self.encoder.forward_pass(x)