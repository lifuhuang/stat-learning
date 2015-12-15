# -*- coding: utf-8 -*-
"""
A SVM implementation based on SMO algorithm, which is invented by John Platt 
from Microsoft Research. Please see his paper for reference.

@author: Lifu Huang
"""

import numpy as np
import matplotlib.pyplot as plt
import functools

class lifsvm:
    eps = 1e-7
    tol = 1e-3
    def __init__(self, X_train, y_train, C, kernel = None):
        '''Initializes a new lifsvm trained on given training set and parameters.
        '''
        self.kernel = kernel if kernel else self.linear_kernel
        self.c = C
        self.m, self.n = X_train.shape
        self.point = X_train
        self.target = y_train
        self.alpha = np.zeros((self.m, 1))
        self.b = 0
        self.error_cache = np.empty((self.m, 1))
        self.main_routine()
                
    def get_error(self, i):
        '''Returns the value of Ei, which equals the SVM output on X[i] minus y[i].
        '''
        self.error_cache[i] = self.hypothesis(self.point[i, :, np.newaxis]) - self.target[i]
        return self.error_cache[i]
        
    def take_step(self, i1, i2):
        '''Makes joint optimization on alpha[i1] and alpha[i2].
        '''
        if i1 == i2:
            return False
        alpha1, alpha2 = self.alpha[i1], self.alpha[i2]
        y1, y2 = self.target[i1], self.target[i2]
        s = y1 * y2
        if s == 1:
            l, h = max(0, alpha2 + alpha1 - self.c), min(self.c, alpha2 + alpha1)
        else:
            l, h = max(0, alpha2 - alpha1), min(self.c, self.c + alpha2 - alpha1)                     
        if l == h:
            return False             
        e1, e2 = self.get_error(i1), self.get_error(i2)
        k11 = self.kernel(self.point[i1, :, np.newaxis], self.point[i1, :, np.newaxis])
        k22 = self.kernel(self.point[i2, :, np.newaxis], self.point[i2, :, np.newaxis])
        k12 = self.kernel(self.point[i1, :, np.newaxis], self.point[i2, :, np.newaxis])
        eta = k11 + k22 - 2 * k12
        if eta > 0 + self.eps:
            a2 = self.clip(alpha2 + y2 * (e1 - e2) / eta, l, h)
        else:
            #Case when eta < 0 (happens when kernel matrix is indefinite)
            #       or eta = 0 (happens when phi(X[i1]) equals phi(X[i2]))
            f1 = y1 * (e1 - b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (e2 - b) - s * alpha1 * k12 - alpha2 * k22
            l1 = alpha1 + s * (alpha2 - l)
            h1 = alpha1 + s * (alpha2 - h)
            objL = l1 * f1 + l * f2 + 0.5 * l1 * l1 * k11 + 0.5 * l * l * k22 + s * l * l1 * k12
            objH = h1 * f1 + h * f2 + 0.5 * h1 * h1 * k11 + 0.5 * h * h * k22 + s * h * h1 * k12
            if objL < objH - self.eps:
                a2 = l
            elif objL > objH + self.eps:
                a2 = h
            else:
                a2 = alpha2
        if np.abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
            return False
        a1 =  alpha1 + s * (alpha2 - a2)
        b1 = float(-e1 - y1 * k11 * (a1 - alpha1) - y2 * k12 * (a2 - alpha2) + self.b)
        b2 = float(-e2 - y1 * k12 * (a1 - alpha1) - y2 * k22 * (a2 - alpha2) + self.b)
        self.b = (b1 + b2) / 2
        self.alpha[i1], self.alpha[i2] = a1, a2
        return True
        
    def examine_example(self, i2):
        '''Inner loop of SMO.
        '''
        e2 = self.get_error(i2)
        r2 = e2 * self.target[i2]
        if (r2 < -self.tol and self.alpha[i2] < self.c - self.eps) or (r2 > self.tol and self.alpha[i2] > 0 + self.eps):
            i1 = max((i for i in range(m) if i != i2), key = lambda i: np.abs(self.error_cache[i] - e2))
            if self.take_step(i1, i2):
                return True
            start = np.random.randint(0, m)
            for delta in range(m):
                i1 = (start + delta) % m
                if i1 != i2 and 0 + self.eps < self.alpha[i1] < self.c - self.eps and self.take_step(i1, i2):
                    return True
            start = np.random.randint(0, m)
            for delta in range(m):
                i1 = (start + delta) % m
                if i1 != i2 and not (0 + self.eps < self.alpha[i1] < self.c - self.eps) and self.take_step(i1, i2):
                    return True    
        return False
    
    def main_routine(self):
        '''Outer loop of SMO
        '''
        examine_all = True
        modified = False
        times = 0
        while modified or examine_all:
            modified = 0
            for i in range(self.m):
                if examine_all or (0 + self.eps < self.alpha[i] < self.c - self.eps):
                    if self.examine_example(i):
                        modified += 1                        
            print('iteration #', times, 'examine_all:', examine_all, 'modified: ', modified) 
            examine_all = not examine_all and modified == 0
            times += 1
            
    def get_precision(self):
        '''Returns the precision on training set.
        '''
        cnt = 0
        for i in range(self.m):
            result = 1 if self.hypothesis(self.point[i, :, np.newaxis]) >= 0 else -1
            if result == self.target[i]:
                cnt += 1
        return cnt / m
        
    def hypothesis(self, x):    
        '''Outputs predicted value on point x.
        '''
        ans = float(self.b)
        for i in range(self.m):
            ans += float(self.alpha[i] * self.target[i] * self.kernel(self.point[i, :, np.newaxis], x))
        return ans
    
    def __call__(self, x):
        return self.hypothesis(x)
    
    @staticmethod
    def linear_kernel(x, z):
        return x.T @ z        
        
    @staticmethod
    def gaussian_kernel(x, z, tau):
        return np.exp(- np.linalg.norm(x - z) ** 2 / tau ** 2)

    @staticmethod
    def clip(val, l, h):
        return max(min(val, h), l)
    
if __name__ == '__main__':    
    X_train = np.fromfile('data/x.dat', sep = ' ').reshape(-1, 2)
    y_train = np.array(list(map(lambda x: x * 2 - 1, np.fromfile('data/y.dat', sep = ' ')))).reshape(-1, 1)
    
    m, n = X_train.shape
    kernel = lifsvm.linear_kernel
    #kernel = functools.partial(lifsvm.gaussian_kernel, tau = 0.25)
    svm = lifsvm(X_train, y_train, 1, kernel)
    
    cnt = 0
    for i in range(m):
        result = 1 if svm(X_train[i, :, np.newaxis]) >= 0 else -1
        if result == y_train[i]:
            cnt += 1
    print('Precision:', cnt / m)
    
    for i in range(m):
        h = svm(X_train[i, :, np.newaxis])
        color = 'r' if y_train[i] == 1 else 'b'
        if abs(y_train[i] * h - 1) <= svm.tol:
            plt.plot(X_train[i, 0], X_train[i, 1], 'D' + color)
        elif y_train[i] * h < 1 - svm.tol:
            plt.plot(X_train[i, 0], X_train[i, 1], 'x' + color)
        elif y_train[i] * h > 1 + svm.tol:
            plt.plot(X_train[i, 0], X_train[i, 1], 'o' + color)
    
    sz = 50
    X, Y = np.meshgrid(np.linspace(-1, 1, sz), np.linspace(-1, 1, sz))
    Z = np.empty((sz, sz))
    for i in range(sz):
        for j in range(sz):
            Z[i, j] = svm(np.array([[X[i, j]], [Y[i, j]]]))
    
    plt.contourf(X, Y, Z, levels= [-1, 0, 1])
    plt.show()
    