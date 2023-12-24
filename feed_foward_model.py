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
        '''
        Variables:
            self.L: number of hidden layers + 1 (output layer)
            self.store: Storing weight and bias of the saved MLP
        '''
        self.L = None
        self.store = None

    def fit(self, store, L):
        '''
        METHOD DESCRIPTION:
            Save the weights and number of hidden layers to Model class
        
        Variables:
            store: Weight dictionary 
            L: number of hidden layers + 1 (output layer)
        '''
        self.store = store
        self.L = L

    def fowardProp(self, X):
        '''
        METHOD DESCRIPTION:
            Given an instance of input vector X, feedfoward X through
            Neural network to output layer.
        
        Variables:
            X: an instance of input vector    
        Return:
            The output layer A[Final_Layer]
        '''
        self.store["A0"] = X.T

        for layer in range(1, self.L + 1):
            
            #Z[layer] = W[layer - 1] * A[layer - 1] + B[layer]
            self.store["Z" + str(layer)] = np.dot(self.store["W" + str(layer)], self.store["A" + str(layer - 1)]) + self.store["b" + str(layer)]
            # (L, L - 1) x (L - 1, m) = (L, m)
            
            #A[layer] = activation(Z[layer])
            if layer != self.L:
                self.store["A" + str(layer)] = leakyRelu(self.store["Z" + str(layer)])
            else:
                self.store["A" + str(layer)] = softmax(self.store["Z" + str(layer)])

        return self.store["A" + str(self.L)]
    
    def predict(self, X):
        '''
        METHOD DESCRIPTION:
            Predict the label of an instance vector X.
        
        Variables:
            X: an instance vecotr
        
        Return:
            Return predicted label with probability.
        '''
        A_L = self.fowardProp(X)
        y_hat = np.argmax(A_L, axis = 0) # (L, m) to (1, m)
        prob = np.max(A_L, axis = 0)
        return y_hat, prob