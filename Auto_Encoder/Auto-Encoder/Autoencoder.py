import numpy as np
import os
import shutil
import json
from layers import Layers
from sequential import Sequential
from optimizer import Adam, SGD
from metrics import MSELoss, Binary_Cross_Entropy

class Encoder:
    def __init__(self, latent_dim = 16):
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()

    def build_encoder(self):
        encoder = Sequential()
        encoder.add(Layers.Convo(num_filter=16, kernel_size=(3, 3), activation='relu', stride=1, padding='same', input_shape=(1, 28, 28, 1), initialize='he'))
        encoder.add(Layers.MaxPooling(pool_size=(2, 2), stride=2))
        encoder.add(Layers.Convo(num_filter=32, kernel_size=(3, 3), activation='relu', stride=1, padding='same', input_shape=(1, 14, 14, 16)))
        encoder.add(Layers.MaxPooling(pool_size=(2, 2), stride=2))
        encoder.add(Layers.Flatten())
        encoder.add(Layers.Dense(dim=(1568, self.latent_dim), activation='relu', train_bias=False, initialize='he'))
        return encoder

    def forward(self, x):
        return self.encoder.forward_pass(x)
    
    def backward(self, grad_z):
        self.encoder.backpropagation(y_true=grad_z)
        # for k, v in self.decoder._layers[0].backward_cache.items():
        #     print(k, v.shape)
        return self.encoder._layers[0].backward_cache['dZ']

    
class Decoder:
    def __init__(self, latent_dim = 16):
        self.latent_dim = latent_dim
        self.decoder = self.build_decoder()
    
    def build_decoder(self):
        decoder = Sequential()
        decoder.add(Layers.Dense(dim=(self.latent_dim, 1568), activation="relu", train_bias=False, initialize='he'))
        decoder.add(Layers.UnFlatten(target_shape=(1, 7, 7, 32)))
        decoder.add(Layers.Upsample(size=(2, 2)))
        decoder.add(Layers.ConvoTranspose(num_filter=16, kernel_size=(3, 3), activation='relu', stride=1, padding='same', input_shape=(1, 14, 14, 32), initialize='he'))
        decoder.add(Layers.Upsample(size=(2, 2)))
        decoder.add(Layers.ConvoTranspose(num_filter=1, kernel_size=(3, 3), activation='sigmoid', stride=1, padding='same', input_shape=(1, 28, 28, 16), initialize='xavier'))
        return decoder  
    
    def forward(self, x):
        return self.decoder.forward_pass(x)
    
    def backward(self, grad_recon):
        self.decoder.backpropagation(y_true=grad_recon)
        # for k, v in self.decoder._layers[0].backward_cache.items():
        #     print(k, v.shape)
        return self.decoder._layers[0].backward_cache['dZ']
        
class AutoEncoder:
    
    def __init__(self, latent_dim=16, optimizer = 'sgd', metric = 'mse', train_bias=False):
        self.latent_dim = latent_dim
        self.train_bias = train_bias
        self.autoencoder = self.build_autoencoder()
        if metric in ['mse', 'bce']:
            self.metric = metric
        else: 
            raise ValueError(f"Expect 'mse'(MSE Loss) or 'bce'(Binary Cross Entropy Loss), found {metric}")
        self.optim = optimizer

    def build_autoencoder(self):
        autoencoder = Sequential()
        # ! encoder 
        autoencoder.add(Layers.Convo(num_filter=16, kernel_size=(3, 3), activation='leakyrelu', stride=1, padding='same', input_shape=(1, 28, 28, 1), initialize='he', train_bias = self.train_bias)) # ? layer : 1
        autoencoder.add(Layers.BatchNorm(num_channels=16, train_bias=self.train_bias))
        autoencoder.add(Layers.MaxPooling(pool_size=(2, 2), stride=2)) # ? layer : 2
        autoencoder.add(Layers.Convo(num_filter=32, kernel_size=(3, 3), activation='leakyrelu', stride=1, padding='same', input_shape=(1, 14, 14, 16), initialize='he', train_bias=self.train_bias)) # ? layer :3
        autoencoder.add(Layers.BatchNorm(num_channels=32, train_bias=self.train_bias))
        autoencoder.add(Layers.MaxPooling(pool_size=(2, 2), stride=2)) # ? layer : 4
        autoencoder.add(Layers.Flatten()) # ? layer : 5
        autoencoder.add(Layers.Dense(dim=(1568, self.latent_dim), activation='leakyrelu', train_bias=self.train_bias, initialize='he')) # ? layer : 6
        # ! decoder 
        autoencoder.add(Layers.Dense(dim=(self.latent_dim, 1568), activation='leakyrelu', train_bias=self.train_bias, initialize='he')) # ? layer : 7
        autoencoder.add(Layers.UnFlatten(target_shape=(1, 7, 7, 32))) # ? layer : 8
        autoencoder.add(Layers.ConvoTranspose(num_filter=16, kernel_size=(3, 3), activation='leakyrelu', stride=2, padding='same', input_shape=(1, 7, 7, 32), initialize='he', train_bias=self.train_bias)) # ? layer : 10
        autoencoder.add(Layers.BatchNorm(num_channels=16, train_bias=self.train_bias))
        autoencoder.add(Layers.ConvoTranspose(num_filter=8, kernel_size=(3, 3), activation='leakyrelu', stride=2, padding='same', input_shape=(1, 14, 14, 16), initialize='he', train_bias=self.train_bias)) # ? layer : 12
        autoencoder.add(Layers.BatchNorm(num_channels=8, train_bias=self.train_bias))
        autoencoder.add(Layers.Convo(num_filter=1, kernel_size=(3, 3), activation='sigmoid', stride=1, padding='same', input_shape=(1, 28, 28, 8), initialize='xavier', train_bias=self.train_bias))
        
        return autoencoder

    def fit(self, x_train, epochs=1000, learning_rate=1e-3, batch_size=16, loss_per_epochs = 1, regularization=None, lambda_reg=0.01, check_point=False):
        from tqdm import tqdm
        self.epochs = epochs
        self.check_point = check_point
        if self.metric == 'mse':
            self.loss_fn = MSELoss(regularization=regularization, lambda_reg=lambda_reg)
        elif self.metric == 'bce': 
            self.loss_fn = Binary_Cross_Entropy(regularization=regularization, lambda_reg=lambda_reg)
        self.loss_per_epochs = loss_per_epochs
        if self.optim == 'sgd':
            self.optimizer = SGD(model=self.autoencoder, learning_rate=learning_rate)
        elif self.optim == 'adam':
            self.optimizer = Adam(model= self.autoencoder, learning_rate=learning_rate, beta1=.9, beta2=.999, epsilon=1e-8)

        self.loss_curve = []
        min_loss = 1e10
        loss = 0
        
        if self.check_point == True and os.path.exists('model/check_points'):
            self.clear_checkpoint()
            self.save_hyper_params(file_path='model/check_points/hyperparameters.json', epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, loss_per_epochs=loss_per_epochs, regularization=regularization, lambda_reg=lambda_reg, check_point=check_point)

        # for epoch in tqdm(range(self.epochs), desc='Epochs'):
        for epoch in (pbar := tqdm(range(self.epochs), desc="Epochs")):
            random_idx = np.random.choice(x_train.shape[0], size=batch_size, replace=False)
            for idx in random_idx:
                xi = x_train.iloc[idx, :].to_numpy().reshape(1, 28, 28, 1)

                # forward
                x_reconstructed = self.forward(x=xi)

                # Loss
                loss = self.loss_fn(true_value=xi, predict_value=x_reconstructed, weights=self.autoencoder.parameters())
                if loss < min_loss:
                    min_loss = loss
                    self.best_params, self.best_loss = self._best_loss(loss)

                grad_loss = self.loss_fn.grad_loss(true_value=xi, predict_value=x_reconstructed)
                if self.loss_fn.regularization == 'l1' or self.loss_fn.regularization == 'l2':
                    regularization_term = self.loss_fn.get_regularization_term(weights=self.autoencoder.parameters())
                    # backpropagation
                    self.backward(gradient_loss=grad_loss, regularization_term=regularization_term)
                else:
                    self.backward(gradient_loss=grad_loss)
                # update weights
                self.optimizer.step()
            if (epoch+1) % self.loss_per_epochs == 0:
                self.loss_curve.append(loss)
                if self.check_point == True:
                    self.save_checkpoint(num = epoch+1)
            pbar.set_postfix_str(f"Loss={loss:.4f}")

    def _best_loss(self, loss):
        '''
        This function will store trainable parameters (weights, bias, ...) when the loss is lowest.
        This trainable parameters can be use to load into model.
        '''
        layer_len = self.autoencoder.get_layer_length()
        best_loss = loss
        best_params = [None] * layer_len
        
        for i in range(layer_len):
            if self.autoencoder._layers[i].weights is not None:
                best_params[i] = self.autoencoder._layers[i].weights
        
        return best_params, best_loss

    def clear_checkpoint(self):
        checkpoint_folder = 'model/check_points'
        if os.path.exists(checkpoint_folder):
            shutil.rmtree(checkpoint_folder)  
        os.makedirs(checkpoint_folder)  

    def save_hyper_params(self, file_path, **kwargs):
        hyper_params = {}
        
        for k, v in kwargs.items(): 
            hyper_params[k] = v

        with open(file_path, 'w') as f:
            json.dump(hyper_params, f, indent=4)

    def save_checkpoint(self, parent_path=None, num = 0):
        if parent_path is None: 
            parent_path = 'model/check_points'
        os.makedirs(parent_path, exist_ok=True)
        file_path = os.path.join(parent_path, f'checkpoint_epochs_{num}_weights.npz')
        file_path_ = os.path.join(parent_path, f'check_point_epochs_{num}_bias.npz')
        weights = self.autoencoder.parameters()
        biases = self.autoencoder.intercept()
        weights = [w if isinstance(w, np.ndarray) else np.array(w, dtype=np.float32) for w in weights]
        biases = [b if isinstance(b, np.ndarray) else np.array(b, dtype=np.float32) for b in biases]
        np.savez(file_path, *weights)
        np.savez(file_path_, *biases)

    def save_model(self, file_path = None):
        if file_path is None: 
            parent_path = 'model/parameters'
            os.makedirs(parent_path, exist_ok=True)
            file_path = os.path.join(parent_path, 'model_weights.npz')
            file_path_ = os.path.join(parent_path, 'model_bias.npz')
        weights = self.autoencoder.parameters()
        biases = self.autoencoder.intercept()
        weights = [w if isinstance(w, np.ndarray) else np.array(w, dtype=np.float32) for w in weights]
        biases = [b if isinstance(b, np.ndarray) else np.array(b, dtype=np.float32) for b in biases]
        np.savez(file_path, *weights)
        np.savez(file_path_, *biases)
        print(f"Model parameters is saved to {file_path}")

    def load_model(self, file_path_weights = None, file_path_bias = None):
        if file_path_weights is None or file_path_bias is None:
            parent_path = 'model/parameters'
            os.makedirs(parent_path, exist_ok=True)
            file_path_weights = os.path.join(parent_path, 'model_weights.npz')
            file_path_bias = os.path.join(parent_path, "model_bias.npz")

        if not os.path.exists(file_path_bias) or not os.path.exists(file_path_weights):
            raise FileNotFoundError(f"No saved model found at {file_path}")

        loaded_weights = np.load(file_path)
        weights = [loaded_weights[key] for key in loaded_weights.files]

        restored_weights = []

        loaded_biases = np.load(file_path_)
        biases = [loaded_biases[key] for key in loaded_biases.files]

        restored_biases = []

        for w in weights:
            if w.shape == ():  
                restored_weights.append(0)
            else:
                restored_weights.append(w)
        for b in biases:
            if b.shape == ():
                restored_biases.append(0)
            else : 
                restored_biases.append(b)

        self.load(params=restored_weights, bias=restored_biases)
        print(f"Successfully loaded parameters from {file_path}")

    def load(self, params, bias):
        '''
        This function will load trainable parameters into model. Use to reconstructed immediately without re-trianing the model.
        '''
        for i in range(self.autoencoder.get_layer_length()):
            if isinstance(params[i], np.ndarray) and params[i].shape != ():
                self.autoencoder._layers[i].weights = params[i]
            else:
                self.autoencoder._layers[i].weights = None
            if isinstance(bias[i], np.ndarray) and bias[i].shape != ():
                self.autoencoder._layers[i].bias = bias[i]
            else:
                self.autoencoder._layers[i].bias = None
        

    def predict(self, input_img):
        return self.autoencoder.predict(input_img=input_img)
    
    def forward(self, x):
        return self.autoencoder.forward_pass(x)
    
    def backward(self, gradient_loss, regularization_term=0):
        self.autoencoder.backpropagation(grad_loss = gradient_loss, regularization_term = regularization_term)
