import numpy as np

class MSELoss:
    
    def __init__(self, regularization = None, lambda_reg = 0.01):
        self.regularization = self._valid_regulation(regularization=regularization)
        self.lambda_reg = lambda_reg

    def _valid_regulation(self, regularization):
        if regularization in ['l1', 'l2', None] :
            return regularization
        else: 
            raise ValueError(f"Support l1, l2 regularization, got {self.regularization}")

    def add_regu_term(self, weights=None):
        if self.regularization == 'l2':
            sum_weights = 0
            for i in weights:
                if isinstance(i, np.ndarray):
                    continue
                else:
                    sum_weights += np.sum(i)
            l2_term = self.lambda_reg * sum_weights
            return l2_term
        elif self.regularization == 'l1':
            sum_weights = 0
            for i in weights:
                if isinstance(i, np.ndarray):
                    continue
                else:
                    sum_weights += np.sum(np.abs(i))
            l1_term = self.lambda_reg * sum_weights
            return l1_term

    def loss_function(self, true_value, predict_value):
        return np.mean((true_value - predict_value)**2) 

    def calculateLoss(self, true_value, predict_value, weights=None):
        mse_loss = self.loss_function(true_value=true_value, predict_value=predict_value) 
        if self.regularization == 'l2' or self.regularization == 'l1':
            _add_regu_term = self.add_regu_term(weights = weights)
            return mse_loss + _add_regu_term
        else:
            return mse_loss

    def grad_loss(self, true_value, predict_value):   
        true_value_flat = true_value.flatten()
        predict_value_flat = predict_value.flatten()

        grads = 2 * (predict_value_flat - true_value_flat) / true_value_flat.size
        return grads 

    def get_regularization_term(self, weights = None) -> list:
        if self.regularization == 'l1':
            terms = []
            for w in weights: 
                if isinstance(w, np.ndarray):
                    term = self.lambda_reg * np.sign(w)
                else:
                    term = 0
                terms.append(term)
        else:
            terms = []
            for w in weights: 
                if isinstance(w, np.ndarray):
                    term = self.lambda_reg * 2 * w
                else:
                    term = 0
                terms.append(term)
        return terms

    def __call__(self, true_value, predict_value, weights=None):
        return self.calculateLoss(true_value=true_value, predict_value=predict_value, weights=weights)

class Binary_Cross_Entropy(MSELoss):

    def  __init__(self, regularization=None, lambda_reg=0.01):
        super().__init__(regularization=regularization, lambda_reg=lambda_reg)

    def loss_function(self, true_value, predict_value):
        epsilon = 1e-12
        predict_value=np.clip(predict_value, epsilon, 1-epsilon)
        return np.mean(-(true_value * np.log(predict_value) + (1 - true_value) * np.log(1 - predict_value)))

    def grad_loss(self, true_value, predict_value):
        eps = 1e-12
        predict_value = np.clip(predict_value, eps, 1 - eps)
        grad = -(true_value / predict_value) + ((1 - true_value) / (1 - predict_value))
        return grad / true_value.size  
