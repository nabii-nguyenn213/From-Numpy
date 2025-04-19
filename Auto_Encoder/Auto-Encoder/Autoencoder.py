import numpy as np
from layers import Layers
from sequential import Sequential
from optimizer import Adam, SGD
from metrics import MSELoss

class Encoder:
    def __init__(self, latent_dim = 16):
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()

    def build_encoder(self):
        encoder = Sequential()
        encoder.add(Layers.Convo(num_filter=16, kernel_size=(3, 3), activation='relu', stride=2, padding='same', input_shape=(1, 28, 28, 1)))
        encoder.add(Layers.Convo(num_filter=32, kernel_size=(3, 3), activation='relu', stride=2, padding='same', input_shape=(1, 14, 14, 16)))
        encoder.add(Layers.Flatten())
        encoder.add(Layers.Dense(dim=(1568, self.latent_dim), activation='linear', train_bias=True, xavier_uniform=True))
        return encoder

    def forward(self, x):
        return self.encoder.forward_pass(x)
    
    def backward(self, grad_z):
        return self.encoder.backpropagation(y_true=grad_z)  # pass gradient from decoder pass
    
class Decoder:
    def __init__(self, latent_dim = 16):
        self.latent_dim = latent_dim
        self.decoder = self.build_decoder()
    
    def build_decoder(self):
        decoder = Sequential()
        decoder.add(Layers.Dense(dim=(self.latent_dim, 1568), activation="relu", train_bias=True, xavier_uniform=True))
        decoder.add(Layers.UnFlatten(target_shape=(1, 7, 7, 32)))
        decoder.add(Layers.ConvoTranspose(num_filter=16, kernel_size=(3, 3), activation='relu', stride=2, padding='same', input_shape=(1, 7, 7, 32)))
        decoder.add(Layers.ConvoTranspose(num_filter=1, kernel_size=(3, 3), activation='sigmoid', stride=2, padding='same', input_shape=(1, 14, 14, 32)))
        return decoder  
    
    def forward(self, x):
        return self.decoder.forward_pass(x)
    
    def backward(self, grad_recon):
        self.decoder.backpropagation(y_true=grad_recon)
        # for k, v in self.decoder._layers[0].backward_cache.items():
        #     print(k, v.shape)
        return self.decoder._layers[0].backward_cache['dZ']
        
class AutoEncoder:
    
    def __init__(self, latent_dim=16):
        self.latent_dim = latent_dim
        self.autoencoder = self.build_autoencoder()
        self.loss_fn = MSELoss()
        
    def build_autoencoder(self):
        autoencoder = Sequential()
        # ! encoder 
        autoencoder.add(Layers.Convo(num_filter=16, kernel_size=(3, 3), activation='relu', stride=2, padding='same', input_shape=(1, 28, 28, 1)))
        autoencoder.add(Layers.Convo(num_filter=32, kernel_size=(3, 3), activation='relu', stride=2, padding='same', input_shape=(1, 14, 14, 16)))
        autoencoder.add(Layers.Flatten())
        autoencoder.add(Layers.Dense(dim=(1568, self.latent_dim), activation='linear', train_bias=False, xavier_uniform=True))
        # ! decoder 
        autoencoder.add(Layers.Dense(dim=(self.latent_dim, 1568), activation="relu", train_bias=False, xavier_uniform=True))
        autoencoder.add(Layers.UnFlatten(target_shape=(1, 7, 7, 32)))
        autoencoder.add(Layers.ConvoTranspose(num_filter=16, kernel_size=(3, 3), activation='relu', stride=2, padding='same', input_shape=(1, 7, 7, 32)))
        autoencoder.add(Layers.ConvoTranspose(num_filter=1, kernel_size=(3, 3), activation='sigmoid', stride=2, padding='same', input_shape=(1, 14, 14, 32)))
        
        return autoencoder

    def fit(self, x_train, epochs=1000, learning_rate=1e-3, batch_size=16):
        from tqdm import tqdm
        self.epochs = epochs
        self.optimizer = SGD(model=self.autoencoder, learning_rate=learning_rate)
        self.loss_curve = []

        for epoch in tqdm(range(self.epochs), desc='Epochs'):
            random_idx = np.random.choice(x_train.shape[0], size=batch_size, replace=False)
            for idx in random_idx:
                xi = x_train.iloc[idx, :].to_numpy().reshape(1, 28, 28, 1)

                # forward
                x_reconstructed = self.forward(x=xi)

                # Loss
                loss = self.loss_fn(true_value=xi, predict_value=x_reconstructed)
                grad_loss = 2*(x_reconstructed-xi) / np.prod(xi.shape)

                # backpropagation
                self.backward(gradient_loss=grad_loss)
                # update weights
                self.optimizer.step()
            if epoch % 10 == 0:
                    self.loss_curve.append(loss)
        
    
    def forward(self, x):
        return self.autoencoder.forward_pass(x)
    
    def backward(self, gradient_loss):
        self.autoencoder.backpropagation(gradient_loss)
