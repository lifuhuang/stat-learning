# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:30:52 2016

@author: Leo
"""

import numpy as np
import matplotlib.pyplot as plt

def train(X_train, y_train, _lambda):
    thetaOld, theta = np.ones((n, 1)), np.zeros((n, 1))
    while np.linalg.norm(theta - thetaOld, ord = 2) > 1e-5:
        thetaOld = theta.copy()    
        for i in range(n):
            theta[i] = 0
            Xi = X_train[:, i, np.newaxis]
            thetaip = max((-Xi.T @ (X_train @ theta - y_train) - _lambda) / (Xi.T @ Xi), 0)                
            thetain = min((-Xi.T @ (X_train @ theta - y_train) + _lambda) / (Xi.T @ Xi), 0)
            theta[i] = thetaip
            objp = 0.5 * np.linalg.norm(X_train @ theta - y_train, 2) ** 2 + _lambda * np.linalg.norm(theta, 1)
            theta[i] = thetain
            objn = 0.5 * np.linalg.norm(X_train @ theta - y_train, 2) ** 2 + _lambda * np.linalg.norm(theta, 1)
            if objp < objn:
                theta[i] = thetaip     
    return theta  
    
if __name__ == '__main__':
    _lambda = 0.1
    
    X_train = np.fromfile(r'data\l1ls\x.dat', sep = ' ')
    y_train = np.fromfile(r'data\l1ls\y.dat', sep = ' ')
    m = np.size(y_train, 0)
    n = np.size(X_train) // m

    X_train.shape = (m, n)
    y_train.shape = (m, 1)
    theta = train(X_train, y_train, _lambda)
    print(np.hstack((y_train, X_train @ theta)))