import numpy as np

class Linear_Regression:
    
    def __init__(self, lr = 0.01, epochs = 1000):
        self.lr = lr
        self.epochs = epochs
        
    
    def initialize_parameters(self, features):
        w = np.random.rand(features, 1)
        b = np.random.rand()
        return w, b
          
    def MSE(self, y_true, y_pred):
        '''
        Mean Square Errors : MSE = (1/m) * (np.sum(y_true - y_pred) ** 2)
        '''
        m = y_true.shape[0]
        mse = (1/m) * (np.sum(y_true - y_pred) ** 2)
        return mse
    
    def stochastic_gradient(self, xi, y_true):
        '''
        calculate single point gradient
        '''
        y_pred = np.matmul(self.w.T, xi) + self.b
        error = y_pred - y_true
        dw = np.matmul(xi, error)
        db = np.mean(error)
        return dw, db
    
    def update(self):
        self.w = self.w - self.lr * self.dw
        self.b = self.b - self.lr * self.db
    
    def fit(self, X_train, y_train, batch_size = 32):
        samples, features = X_train.shape
        self.w, self.b = self.initialize_parameters(features)
        self.batch_size = batch_size
        for it in range(self.epochs):
            rand_id = np.random.choice(samples, size = self.batch_size, replace=False)
            for i in rand_id:
                xi = np.array(X_train.iloc[i, :]).reshape(features, 1)
                yi = y_train.iloc[i]
                self.dw, self.db = self.stochastic_gradient(xi, yi)
                self.update()
            
    def predict(self, x):
        return np.matmul(x, self.w) + self.b