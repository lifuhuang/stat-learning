# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
from __init__ import *
    
class DataSet:
    def __init__(self, ratio = (0.8, 0, 0.2)):
        """
        Initializes a Dataset. 
        """        
        self.ratio = ratio
        if sum(self.ratio) != 1:
            raise ValueError("tuple ratio must sum to 1")
        self.cv_cnt = 0
        self.training_cnt = 0
        self.test_cnt = 0
        self.n_samples = 0
        self.n_training_samples = 0
        self.n_test_samples = 0
        self.training_features = None
        self.training_labels = None        
        self.cv_features = None
        self.cv_labels = None
        self.test_features = None
        self.test_labels = None
        self.filled = False
        
        
    def load(self, features, labels):
        """
        Loads data to DataSet.
        """
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Inconcistent sample number!")
        self.n_samples = int(features.shape[0])
        self.n_training_samples = int(self.n_samples * self.ratio[0])
        self.n_cv_samples = int(self.n_samples * self.ratio[1])
        self.n_test_samples = int(self.n_samples - self.n_training_samples - self.n_cv_samples)
        
        shuffled_ids = [i for i in xrange(self.n_samples)]
        random.shuffle(shuffled_ids)
        
        self.training_features = features[shuffled_ids[:self.n_training_samples]]
        self.training_labels = labels[shuffled_ids[:self.n_training_samples]]
        
        self.cv_features = features[shuffled_ids[self.n_training_samples:
            self.n_training_samples+self.n_cv_samples]]
        self.cv_labels = labels[shuffled_ids[self.n_training_samples:
            self.n_training_samples+self.n_cv_samples]]
        
        self.test_features = features[shuffled_ids[-self.n_test_samples:]]
        self.test_labels = labels[shuffled_ids[-self.n_test_samples:]]
        
        self.reset()
        self.filled = True        

        
    def load_from_files(self, feature_file_path, label_file_path):
        """
        Loads data from files.
        """
        self.load(np.fromfile(feature_file_path), np.fromfile(label_file_path))
    
    def check_filled(self):
        """
        Tests whether the DataSet has been filled with data.
        """
        if not self.filled:
            raise Exception("Unfilled dataset, please load data first.")
        
    def reset(self):
        """
        Resets counter to zero.
        """
        self.cv_cnt = 0
        self.training_cnt = 0
        self.test_cnt = 0

    def get_training_sample(self):
        """
        Gets a training sample in cyclic order.
        """
        self.check_filled()
        features = self.training_features[self.training_cnt]
        label = self.training_labels[self.training_cnt]        
        self.training_cnt = (self.training_cnt + 1) % self.n_training_samples
        return features, label        
        
    def get_random_training_sample(self):
        """
        Gets a training sample randomly.
        """
        self.check_filled()
        idx = random.randint(0, self.n_training_samples - 1)
        return self.training_features[idx], self.training_labels[idx]
    
    def get_cv_sample(self):
        """
        Gets a CV sample in cyclic order.
        """
        self.check_filled()
        features = self.cv_features[self.cv_cnt]
        label = self.cv_labels[self.cv_cnt]        
        self.cv_cnt = (self.cv_cnt + 1) % self.n_cv_samples
        return features, label            
        
    def get_random_cv_sample(self):
        """
        Gets a CV sample randomly.
        """
        self.check_filled()
        idx = random.randint(0, self.n_cv_samples - 1)
        return self.cv_features[idx], self.cv_labels[idx]
        
    def get_test_sample(self):
        """
        Gets a test sample in cyclic order.
        """
        self.check_filled()
        features = self.test_features[self.test_cnt]
        label = self.test_labels[self.test_cnt]        
        self.test_cnt = (self.test_cnt + 1) % self.n_test_samples
        return features, label            
        
    def get_random_test_sample(self):
        """
        Gets a test sample randomly.
        """
        self.check_filled()
        idx = random.randint(0, self.n_test_samples - 1)
        return self.test_features[idx], self.test_labels[idx]
            

class SoftmaxNN:   
    def __init__(self, layers):
        """
        Initializes a Softmax neural network. 
        """
        self.n_layers = len(layers)
        self.s_layer = tuple(layers)
        self.w = None
        self.b = None
        self.fitted = False
            
    def sgd_wrapper(self, theta, dataset, randomized = False):
        """
        Wrapper used for stochastic gradient descent.
        """
        self.set_parameters(theta)
        if randomized:
            features, label = dataset.get_random_traing_sample()
        else:
            features, label = dataset.get_training_sample()
            
        target = np.zeros((self.s_layer[-1],))
        target[label] = 1
        
        cost, grad_w, grad_b = self.feed_forward(features, target)
        lst = []        
        for i in xrange(self.n_layers - 1):
            lst.append(grad_w[i].flatten())
            lst.append(grad_b[i].flatten())
        grad = np.concatenate(lst)
        return cost, grad
        
    def check_fitted(self):   
        """
        Checks whether the model has been fitted.
        """
        if not self.fitted:
            raise Exception('Model cannot be used before fitted to a dataset.')
            
            
    def forward_propagate(self, features):
        """
        Calculates the "a" matrices for all layers.
        """
        a = []
        a.append(features)
        for i in xrange(self.n_layers - 1):
            if i < self.n_layers - 2:
                a.append(sigmoid(self.w[i].dot(a[i]) + self.b[i]))
            else:
                a.append(softmax(self.w[i].dot(a[i]) + self.b[i]))
        return a
            
    def back_propagate(self, a, target):
        """
        Calculates the gradients of w and b using BP algorithm.
        """   
        delta = [None for i in xrange(self.n_layers)]
        delta[-1] = a[-1] - target
        for i in xrange(self.n_layers - 2, 0, -1):
            delta[i] = a[i] * (1 - a[i]) * (self.w[i].T.dot(delta[i+1]))
        
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
        """
        Gets the total number of all parameters in this NN.
        """
        return sum((self.s_layer[i+1] * (self.s_layer[i] + 1))
                    for i in xrange(self.n_layers - 1))
            
    def set_parameters(self, theta):
        """
        Sets parameters of this NN from argument theta.
        """
        self.w = []
        self.b = []
        p = 0
        for i in xrange(self.n_layers - 1):
            self.w.append((theta[p:p+self.s_layer[i+1]*self.s_layer[i]]
                            .reshape(self.s_layer[i+1], self.s_layer[i])))
            p += self.s_layer[i+1] * self.s_layer[i]
            self.b.append(theta[p:p+self.s_layer[i+1]])
            p += self.s_layer[i+1]
        
        
    def get_parameters(self):
        """
        Returns a 1D array consist of all parameters in this NN.
        """
        return np.concatenate(sum([w[i].flatten(), b[i].flatten()] 
                                    for i in xrange(self.n_layers-1)))
        
    def fit(self, dataset, n_iters = 1000, randomized = False):
        """
        Fits this NN to given features and labels.
        """        
        theta0 = np.random.randn(self.get_parameter_count())
        sgd(lambda x: self.sgd_wrapper(x, dataset, randomized), theta0, 0.3, 
            n_iters, None, False, PRINT_EVERY=10)
        self.fitted = True
    
    def predict(self, features):
        """
        Returns the probability of input being in each class given features.
        """
        self.check_fitted()
        return self.forward_propagate(features)[-1]
        