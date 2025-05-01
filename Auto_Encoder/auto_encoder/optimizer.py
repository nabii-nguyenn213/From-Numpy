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

class Adam:
    
    def __init__(self, model, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0
    
    def initialize_moments(self):
        '''
        Params :
            weights = [W(1), W(2), ..., W(n)] where W(i) is the weight of the i layer
            biases = [b(1), b(2), ..., b(n)] where b(i) is the bias of the i layer
        Return :
            m_w = [m_w(1), m_w(2), ..., m_w(n)] where m_w(i) is a first moment estimate zeros matrix have the same shape of W(i)
            v_w = [v_w(1), v_w(2), ..., v_w(n)] where v_w(i) is a second moment estimate zeroszeros matrix have the same shape of W(i)
        '''
        self.m_w = [0] * self.model.get_layer_length()
        self.v_w = [0] * self.model.get_layer_length()
        self.m_b = [0] * self.model.get_layer_length()
        self.v_b = [0] * self.model.get_layer_length()

        for i in range(self.model.get_layer_length()):
            if self.model._layers[i].weights is not None:
                self.m_w[i] = np.zeros(self.model._layers[i].weights.shape)
                self.v_w[i] = np.zeros(self.model._layers[i].weights.shape)
            if self.model._layers[i].bias is not None:
                self.m_b[i] = np.zeros(self.model._layers[i].bias.shape)
                self.v_b[i] = np.zeros(self.model._layers[i].bias.shape)

    def step(self):
        if self.m_w is None or self.v_w is None: 
            self.initialize_moments()
        self.t += 1
        updated_weights = [0] * self.model.get_layer_length()
        updated_bias = [0] * self.model.get_layer_length()

        for i in range(self.model.get_layer_length()):
            if self.model._layers[i].weights is not None:
            # ! weights
                # ? mi <- β1 * mi + (1 - β1) * ∇θ 
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * self.model.backward_caches[i]['dW']
                # ? mi_corrected <- mi / (1 - β1**t)
                m_w_corrected = self.m_w[i] / (1 - self.beta1 ** self.t)
                # ? vi <- β2 * vi + (1 - β2) * ∇θ²
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (self.model.backward_caches[i]['dW']**2)
                # ? vi_corrected <- vi / (1 - β2**t)
                v_w_corrected = self.v_w[i] / (1 - self.beta2 ** self.t)
                # ? Update weights : θ <- θ - (lr / vi_corrected**0.5 + ε) * mi_corrected
                updated_weights[i] = self.model._layers[i].weights - (self.learning_rate /(np.sqrt(v_w_corrected) + self.epsilon)) * m_w_corrected
            if self.model._layers[i].bias is not None:
                g_b = self.model._layers[i].backward_cache['db']
                if g_b is None:
                     continue
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * g_b
                m_b_corr = self.m_b[i] / (1 - self.beta1 ** self.t)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (g_b ** 2)
                v_b_corr = self.v_b[i] / (1 - self.beta2 ** self.t)
                updated_bias[i] = self.model._layers[i].bias - (self.learning_rate * m_b_corr/(np.sqrt(v_b_corr)+self.epsilon))
        
        # UPDATE Params

        for i in range(self.model.get_layer_length()):
            if self.model._layers[i].weights is not None:
                self.model._layers[i].weights = updated_weights[i]
            if self.model._layers[i].bias is not None:
                self.model._layers[i].bias = updated_bias[i]

    def reset(self):
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0
