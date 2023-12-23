import numpy as np
import pandas as pd

def leakyRelu(x):
    return np.where(x > 0, x, 0.01 * x)

def softmax(z):
    ez = np.exp(z - np.max(z, axis = 0))
    return ez / np.sum(ez, axis = 0, keepdims = True)
    #return np.exp(z) / np.sum(np.exp(z), axis = 0, keepdims = True)

def leakyRelu_derivative(x):
    return np.where(x > 0, 1, 0.01)

class Model():
    def __init__ (self):
        self.L = None
        self.store = None

    def fit(self, store, L):
        self.store = store
        self.L = L

    def fowardProp(self, X):
        self.store["A0"] = X.T

        for layer in range(1, self.L + 1):
            self.store["Z" + str(layer)] = np.dot(self.store["W" + str(layer)],
                                                  self.store["A" + str(layer - 1)]) + self.store["b" + str(layer)]
            # (L, L - 1) x (L - 1, m) = (L, m)
            if layer != self.L:
                self.store["A" + str(layer)] = leakyRelu(self.store["Z" + str(layer)])
            else:
                self.store["A" + str(layer)] = softmax(self.store["Z" + str(layer)])

        return self.store["A" + str(self.L)]
    def predict_label(self, X, y):

        A_L = self.fowardProp(X)
        y_hat = np.argmax(A_L, axis = 0) # (L, m) to (1, m)
        y_real = np.argmax(y.T, axis = 0) #(10, m) to (1, m)
        accuracy = (y_hat == y_real).mean()
        return accuracy * 100
    def predict(self, X):

        A_L = self.fowardProp(X)
        y_hat = np.argmax(A_L, axis = 0) # (L, m) to (1, m)
        prob = np.max(A_L, axis = 0)
        return y_hat, prob