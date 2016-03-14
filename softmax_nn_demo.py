# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:13:24 2016

@author: lifu
"""


from liflib2.dl import SoftmaxNN
from liflib2.dl import DataSet
import liflib2
import numpy as np

if __name__ == '__main__':
    n_samples = 100
    n_features = 10
    n_classes = 10
    
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, n_classes, size = n_samples)
    dataset = DataSet()
    dataset.load(features, labels)
    nn = SoftmaxNN((n_features, 50, 50, 50, n_classes))
    nn.fit(dataset, 20000, False)
    theta = np.random.randn(nn.get_parameter_count())
    
    dataset.reset()
    n_correct = 0
    for i in xrange(dataset.n_training_samples):
        features, label = dataset.get_training_sample()
        if np.argmax(nn.predict(features)) == label:
            n_correct += 1
            
    print 'For test set: total: %d, correct: %d, accuracy = %g%%' % (dataset.n_training_samples, n_correct, n_correct * 100.0 / dataset.n_training_samples)
            
    
    dataset.reset()
    n_correct = 0
    
    for i in xrange(dataset.n_test_samples):
        features, label = dataset.get_test_sample()
        if np.argmax(nn.predict(features)) == label:
            n_correct += 1
            
    print 'For test set: total: %d, correct: %d, accuracy = %g%%' % (dataset.n_test_samples, n_correct, n_correct * 100.0 / dataset.n_test_samples)

