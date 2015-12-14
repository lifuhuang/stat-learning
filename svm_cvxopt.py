# -*- coding: utf-8 -*-
"""
Support Vector Machine implemented with off-the-shelf convex optimization.

@author Lifu Huang
"""

import numpy as np
import cvxopt as co
import matplotlib.pyplot as plt
import functools
import itertools

eps = 1e-6
tol = 1e-3

def gaussian_kernel(x, z, tau):
    return np.exp(- np.linalg.norm(x - z) ** 2 / tau ** 2)

def linear_kernel(x, z):
    return x.T @ z
 
def train_svm_with_cvxopt(X, y, C, kernel = linear_kernel):
    m, n = X.shape
    K = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = kernel(X[i, :, np.newaxis], X[j, :, np.newaxis])
    P = (y @ y.T) * K
    q = -np.ones((m, 1))
    G = np.vstack((np.identity(m), -np.identity(m)))
    h = np.vstack((C * np.ones((m, 1)), np.zeros((m, 1))))
    A = y.T
    b = np.zeros((1, 1))    
    res = co.solvers.qp(co.matrix(P), co.matrix(q), co.matrix(G), co.matrix(h), co.matrix(A), co.matrix(b))    
    alpha = np.array(res['x'])
    w = np.sum(alpha * y * X, axis = 0)[:, np.newaxis]
    bSum = np.array([[.0]])
    for j in range(m):
        if 0 + eps < alpha[j] < C - eps:
            bSum += y[j]
            for i in range(m):
                bSum -= y[i] * alpha[i] * kernel(X[i, :, np.newaxis], X[j, :, np.newaxis])
    b = bSum / sum(1 for i in range(m) if 0 + eps < alpha[i] < C - eps)
    return (alpha, w, b)

def predict(x, X_train, y_train, alpha, b, kerneo = linear_kernel):
    m = np.size(X_train, 0)
    n = np.size(X_train, 1)
    ans = float(b)  
    for i in range(m):
        ans += float(alpha[i] * y_train[i] * kernel(X_train[i, :, np.newaxis], x))
    return ans
    
if __name__ == '__main__':    
    X_train = np.fromfile('data/x.dat', sep = ' ').reshape(-1, 2)
    y_train = np.array(list(map(lambda x: x * 2 - 1, np.fromfile('data/y.dat', sep = ' ')))).reshape(-1, 1)
    
    m, n = X_train.shape
    kernel = linear_kernel#functools.partial(gaussian_kernel, tau = 0.25)
    alpha, w, b = train_svm_with_cvxopt(X_train, y_train, 1, kernel)
    print('w:', w, 'b:', b, sep = '\n')
    cnt = 0
    for i in range(m):
        result = 1 if (predict(X_train[i, :, np.newaxis], X_train, y_train, alpha, b, kernel) >= 0) else -1
        if result == y_train[i]:
            cnt += 1
    print('Precision:', cnt / m)
    
    for i in range(m):
        h = predict(X_train[i, :, np.newaxis], X_train, y_train, alpha, b, kernel)
        color = 'r' if y_train[i] == 1 else 'b'
        if abs(y_train[i] * h - 1) <= tol:
            plt.plot(X_train[i, 0], X_train[i, 1], 'D' + color)
        elif y_train[i] * h < 1 - tol:
            plt.plot(X_train[i, 0], X_train[i, 1], 'x' + color)
        elif y_train[i] * h > 1 + tol:
            plt.plot(X_train[i, 0], X_train[i, 1], 'o' + color)
            
    sz = 50
    X, Y = np.meshgrid(np.linspace(-1, 1, sz), np.linspace(-1, 1, sz))
    Z = np.empty((sz, sz))
    for i in range(sz):
        for j in range(sz):
            Z[i, j] = predict(np.array([[X[i, j]], [Y[i, j]]]), X_train, y_train, alpha, b, kernel)
    
    plt.contourf(X, Y, Z, levels = [-1, 0, 1])
    plt.show()
    