# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:01:34 2016

@author: lifu
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

class GDAModel:
    @staticmethod
    def gaussian(x, mu, sig):
        x = x.reshape(-1, 1)
        return (1.0 / ((2 * math.pi) ** (x.size / 2) * np.linalg.det(sig) **
            0.5) * np.exp(-0.5 * (x - mu).T @ np.linalg.inv(sig) @ (x - mu)))

    def __init__(self):
        self.fitted = False
        self.phi = None
        self.mu = None
        self.sigma = None
        self.n_classes = None
        
    def check_fitted(self):        
        if not self.fitted:
            raise Exception('Model cannot be used before fitted to a dataset.')

    def fit(self, data, labels, n_classes):
        n_samples, n_dim = data.shape
        self.n_classes = n_classes
        self.phi = np.empty((self.n_classes, 1))
        self.mu = np.empty((self.n_classes, n_dim, 1))
        self.sigma = np.empty((self.n_classes, n_dim, n_dim))
        
        for k in range(self.n_classes):
            data_k = data[labels.flatten() == k, :]
            self.phi[k] = data_k.shape[0] / n_samples
            self.mu[k] = np.sum(data_k, axis = 0).reshape(-1, 1) / data_k.shape[0]
            self.sigma[k] = (data_k.T @ data_k) / data_k.shape[0]
            
        self.fitted = True
        
    def classify(self, point):
        self.check_fitted()
        return max(range(self.n_classes), 
                   key = lambda k: self.gaussian(point, self.mu[k], self.sigma[k]))
              
    def plot(self):
        self.check_fitted()
        
    
if __name__ == '__main__':
    data_file_path = 'data/GDA/x.dat'
    label_file_path = 'data/GDA/y.dat'
    
    data = np.fromfile(data_file_path, sep = ' ').reshape(-1, 2)
    labels = np.fromfile(label_file_path, sep = ' ').reshape(-1, 1)
    print(data.shape, labels.shape)
    g = GDAModel()
    g.fit(data, labels, 5)
    
    m, n = data.shape
    n_correct = 0
    for i in range(m):
        res = g.classify(data[i].reshape(-1, 1))
        print('testing sample #%d, predicted: %d, real label: %d' % (i, res, labels[i]))
        if res == labels[i]:
            n_correct += 1
    
    print('Done! Correct: Accuracy on training set: %d %%' % (n_correct * 100.0 / m))