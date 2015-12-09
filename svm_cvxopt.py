# -*- coding: utf-8 -*-
"""
Support Vector Machine implemented with convex optimization method
"""

import numpy as np
import cvxopt as co
import matplotlib as mpl
X = np.fromfile('x.dat', sep = ' ').reshape((-1, 2))
y = np.array(list(map(lambda x: x * 2 - 1, np.fromfile('y.dat', sep = ' ')))).reshape((-1, 1))

m = np.size(X, 0)
n = np.size(X, 1)
C = 0.035

P = (X @ X.T) * (y @ y.T)
q = -np.ones(y.shape)
G = np.vstack((np.identity(m), -np.identity(m)))
h = np.vstack((C * np.ones((m, 1)), np.zeros((m, 1))))
A = y.T
b = np.ones((1, 1))

res = co.solvers.qp(co.matrix(P), co.matrix(q), co.matrix(G), co.matrix(h), co.matrix(A), co.matrix(b))

alpha = np.array(res['x'])
w = np.sum(alpha * y * X, axis = 0).reshape((-1, 1))
b = - (w.T @ X[max(filter(lambda i: y[i] == -1, range(m)), key = lambda i: w.T @ X[i])] + 
       w.T @ X[min(filter(lambda i: y[i] == 1, range(m)), key = lambda i: w.T @ X[i])]) / 2

cnt = 0
for i in range(m):
    result = 1 if (w.T @ X[i] + b) > 0 else -1
    if result == y[i]:
        cnt += 1
print('Precision:', cnt / m)

