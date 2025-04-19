import numpy as np

class MSELoss:
    def __init__(self):
        pass

    def calculateLoss(self, true_value, predict_value):
        return np.mean((true_value-predict_value)**2)

    def __call__(self, true_value, predict_value):
        return self.calculateLoss(true_value=true_value, predict_value=predict_value)
