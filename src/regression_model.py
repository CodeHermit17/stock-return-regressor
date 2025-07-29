import numpy as np

class LinearRegressionScratch:

    def __init__(self,learning_rate: float, epochs: int):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.theta=None

    def add_bias(self,X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Ensures X is 2D
        ones_col = np.ones((X.shape[0], 1))
        X_bias=np.hstack((ones_col,X))

        return X_bias
    
    def forward(self,X):
        X_bias = self.add_bias(X)
        y_pred = X_bias @ self.theta

        return y_pred
    
    def compute_loss(self,y,y_pred):
        
        loss = np.mean((y_pred - y) ** 2)
        return loss
    
    def backward(self, X, y,y_pred):

        m=len(y)
        X_bias= self.add_bias(X)
        X_transpose = X_bias.transpose()
        grad_theta = (X_transpose @ (y_pred - y) ) / m
        
        return grad_theta

    def fit(self, X, y, X_test=None, y_test=None):
        self.losses = []
        X_bias=self.add_bias(X)
        num_columns = X_bias.shape[1]
        self.theta=np.zeros((num_columns,))

        for i in range(self.epochs):
            y_pred=self.forward(X)
            loss=self.compute_loss(y,y_pred)
            grad_theta=self.backward(X,y,y_pred)
            self.theta-= self.learning_rate*grad_theta

            self.losses.append(loss)

    def predict(self,X):
        X_bias = self.add_bias(X)
        y_pred = X_bias @ self.theta

        return y_pred