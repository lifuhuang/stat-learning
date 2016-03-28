# -*- coding: utf-8 -*-
'''
Spyder Editor

This is a temporary script file.
'''

import numpy as np
import math
from __init__ import *

class SoftmaxNN:   
    def __init__(self, layers):
        '''
        Initializes a Softmax neural network. 
        '''
        self.n_layers = len(layers)
        self.layer_size = tuple(layers)
        self.w = [None] * (self.n_layers - 1)
        self.b = [None] * (self.n_layers - 1)
        self.fitted = False

    def random_init(self, weight_filler = 'xavier', bias_filler = 'constant'):
        for i in xrange(self.n_layers - 1):
            if weight_filler == 'xavier':        
                r = math.sqrt(6.0 / (self.layer_size[i] 
                                    + self.layer_size[i+1]))
                self.w[i] = np.random.uniform(-r, r, (self.layer_size[i+1], 
                                                        self.layer_size[i]))
            elif weight_filler == 'constant':
                self.w[i] = np.zeros((self.layer_size[i+1], 
                                        self.layer_size[i]))
            elif weight_filler == 'randn':
                self.w[i] = np.random.randn(self.layer_size[i+1], 
                                              self.layer_size[i])
            else:
                raise ValueError('%s is not a valid weight_filler.' % 
                                    weight_filler)
        
            if bias_filler == 'constant':
                self.b[i] = np.zeros(self.layer_size[i+1])
            elif bias_filler == 'randn':
                self.b[i] = np.random.randn(self.layer_size[i+1])
            else:
                raise ValueError('%s is not a valid bias_filler.' % 
                                    bias_filler)
        
                                            
    def objective(self, theta, dataset, batch_size = 50, randomized = True,
                  regularization = None, _lambda = 0):
        '''
        Wrapper used for stochastic gradient descent.
        '''
        self.set_parameters(theta)
        cost = 0
        grad = np.zeros(self.get_parameter_count())
        for i in xrange(batch_size):
            if randomized:
                features, label = dataset.get_random_training_sample()
            else:
                features, label = dataset.get_training_sample()
                
            target = np.zeros((self.layer_size[-1],))
            target[label] = 1
            
            sp_cost, grad_w, grad_b = self.feed_forward(features, target)
            lst = []        
            for i in xrange(self.n_layers - 1):
                lst.append(grad_w[i].flatten())
                lst.append(grad_b[i].flatten())
            cost += sp_cost / float(batch_size)
            grad += np.concatenate(lst) / float(batch_size)
        if regularization:
            regularization = regularization.lower()
            if regularization == 'l2':
                cost += 0.5 * _lambda * np.sum(theta ** 2)
                grad += _lambda * theta
            else:
                raise ValueError('Unknown regularization: %s' % regularization)
        return cost, grad
        
    def check_fitted(self):   
        '''
        Checks whether the model has been fitted.
        '''
        if not self.fitted:
            raise Exception('Model cannot be used before fitted to a dataset.')
            
            
    def forward_propagate(self, features):
        '''
        Calculates the "a" matrices for all layers.
        '''
        a = []
        a.append(features)
        for i in xrange(self.n_layers - 1):
            if i < self.n_layers - 2:
                a.append(np.tanh(self.w[i].dot(a[i]) + self.b[i]))
            else:
                a.append(softmax(self.w[i].dot(a[i]) + self.b[i]))
        return a
            
    def back_propagate(self, a, target):
        '''
        Calculates the gradients of w and b using BP algorithm.
        '''   
        delta = [None] * self.n_layers
        delta[-1] = a[-1] - target
        for i in xrange(self.n_layers - 2, 0, -1):
            delta[i] = (1 - a[i] ** 2) * (self.w[i].T.dot(delta[i+1]))
        
        grad_w = [np.outer(delta[i+1], a[i]) 
                    for i in xrange(self.n_layers - 1)]
        grad_b = [delta[i+1] for i in xrange(self.n_layers - 1)]
        return grad_w, grad_b
    
    def feed_forward(self, features, target): 
        a = self.forward_propagate(features)
        cost = - np.log(np.dot(a[-1], target))
        grad_w, grad_b = self.back_propagate(a, target)
        return cost, grad_w, grad_b
    
    def get_parameter_count(self):
        '''
        Gets the total number of all parameters in this NN.
        '''
        return sum((self.w[i].size + self.b[i].size)
                    for i in xrange(self.n_layers - 1))
            
    def set_parameters(self, theta):
        '''
        Sets parameters of this NN from argument theta.
        '''
        p = 0
        for i in xrange(self.n_layers - 1):
            self.w[i] = (theta[p:p+self.layer_size[i+1] * self.layer_size[i]]
                            .reshape(self.layer_size[i+1], self.layer_size[i]))
            p += self.layer_size[i+1] * self.layer_size[i]
            self.b[i] = theta[p:p+self.layer_size[i+1]]
            p += self.layer_size[i+1]
        
    def get_parameters(self):
        '''
        Returns a 1D array consist of all parameters in this NN.
        '''
        lst = []   
        for i in xrange(self.n_layers - 1):
            lst.append(self.w[i].flatten())
            lst.append(self.b[i].flatten())
        return np.concatenate(lst)
        
    def fit(self, dataset, 
            f_min = sgd, 
            init_options = {}, 
            obj_options = {}, 
            f_min_options = {}):
        '''
        Fits this NN to given features and labels.
        '''      
        self.random_init(**init_options)
        theta = f_min(lambda x: 
            self.objective(x, dataset, **obj_options), 
            self.get_parameters(), 
            **f_min_options)
        self.set_parameters(theta)                  
        self.fitted = True
    
    def predict(self, features):
        '''
        Returns the probability of input being in each class given features.
        '''
        self.check_fitted()
        return self.forward_propagate(features)[-1]
