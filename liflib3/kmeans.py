# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:29:21 2016

@author: Leo
"""

import numpy as np
import matplotlib.pyplot as plt

def k_means(points, k):
    m, n = np.shape(points)
    centroidsOld = np.zeros((k, n))
    centroids = points[np.random.random_integers(0, m, k)]
    clusters = np.empty((m), dtype=int)
    while np.linalg.norm(centroids - centroidsOld, 2) > 1e-15:
        centroidsOld = centroids.copy()
        for i in range(m):
            clusters[i] = min(range(k), key = lambda j: np.linalg.norm(points[i] - centroids[j], 2))
        for i in range(k):
            centroids[i] = np.average(points[clusters == i], axis = 0)
    return centroids, clusters
    
if __name__ == '__main__':
    X_train = np.fromfile('data/kmeans/x.dat', sep = ' ').reshape(-1, 2)
    m, n = X_train.shape
    for i in range(m):
        plt.plot(X_train[i, 0], X_train[i, 1], 'kx')
    plt.show()
    
    centroids, clusters = k_means(X_train, 3)
    print('Done!')
    markers = ['bx', 'rx', 'gx', 'yx']
    
    for i in range(m):
        plt.plot(X_train[i, 0], X_train[i, 1], markers[clusters[i]])
    for centroid in centroids:
        plt.plot(centroid[0], centroid[1], 'ko')
    plt.show()
    