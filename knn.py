from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.utils import to_categorical

import numpy as np
import pandas as pd
import os 



#Gauss kernel weight
def gauss(distance):
    sigma_square = 4
    return np.exp(-(distance**2) / sigma_square)

#KNN model
class KNNModel:
    def __init__(self, neighbor):
            self.knn = KNeighborsClassifier(n_neighbors = neighbor, weights=gauss)
            
    def fit(self):
        # Loading data from MNIST dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        #Preprocessing MNIST
        X_train_reshaped = X_train.reshape(X_train.shape[0], 784) / 255.0
        X_test_reshaped = X_test.reshape(X_test.shape[0], 784) / 255.0
        
        #Concatenating into one set.
        X = np.concatenate((X_train_reshaped, X_test_reshaped), axis = 0)
        y = np.concatenate((y_train, y_test), axis = 0)
        
        #Put it in KNN
        self.knn.fit(X, y)
        
    def predict(self, x_test):
        return self.knn.predict(x_test)
    




