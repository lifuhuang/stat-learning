"""
Mixture of Gaussians model implemented by EM algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def gaussian(x, mu, sig):
    return float(1 / ((2 * math.pi) ** (x.size / 2) * np.linalg.det(sig) ** 0.5) * np.exp(-0.5 * (x - mu).T @ np.linalg.inv(sig) @ (x - mu)))

def GoF(X, k):
    m, n = X.shape
    phiOld, phi = np.zeros((k, 1)), np.ones((k, 1)) / k
    muOld, mu = np.zeros((k, n, 1)), np.random.random((k, n, 1))
    sigOld, sig = np.zeros((k, n, n)), np.empty((k, n, n))
    
    for i in range(k):
        sig[i] = np.diag(np.random.random(n))
    it = 0
    while (np.any(np.linalg.norm(phi - phiOld, ord = 2, axis = 1) > 1e-7) or
            np.any(np.linalg.norm(mu - muOld, ord = 2, axis = (1, 2)) > 1e-7) or
            np.any(np.linalg.norm(sig - sigOld, ord = 2, axis = (1, 2)) > 1e-7)):
        it += 1
        # print('iteration', it)
        # print('sig', sig)
        # print('mu', mu)
        # print('phi', phi)
            
        # E-step
        phiOld = phi.copy()
        muOld = mu.copy()
        sigOld = sig.copy()
        w = np.empty((m, k))
        for i in range(m):
            wiSum = 0
            for j in range(k):
                w[i, j] = gaussian(X[i, :, np.newaxis], mu[j], sig[j]) * phi[j]
                wiSum += w[i, j]
            w[i, :] /= wiSum

        # M-step
        for j in range(k):
            wjSum = np.sum(w[:, j])
            phi[j] = wjSum / m
            mu[j] = np.sum(w[:, j, np.newaxis] * X, axis = 0)[:, np.newaxis] / wjSum
            sig[j] = 0
            for i in range(m):
                sig[j] += w[i, j] * (X[i, :, np.newaxis] - mu[j]) @ (X[i, :, np.newaxis] - mu[j]).T
            sig[j] /= wjSum
    return phi, mu, sig    

if __name__ == '__main__':
    X_train = np.fromfile(r'data\MoG\x.dat', sep = ' ').reshape(-1, 2)
    m, n = X_train.shape
    k = 2
    
    for i in range(m):
        plt.plot(X_train[i, 0], X_train[i, 1], 'rx')
    plt.show()
    
    phi, mu, sig = GoF(X_train, k)
    sz = 100
    X, Y = np.meshgrid(np.linspace(-6, 8, sz), np.linspace(-2, 12, sz))
    Z = np.empty((sz, sz))
    for i in range(sz):
        for j in range(sz):
            Z[i, j] = sum(gaussian(np.array([[X[i, j]], [Y[i, j]]]), mu[l], sig[l]) * phi[l] for l in range(k))
    plt.contourf(X, Y, Z, 100)
    plt.show()