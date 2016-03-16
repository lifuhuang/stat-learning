# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:13:24 2016

@author: lifu
"""

from liflib2.dl import SoftmaxNN
from liflib2.dl import DataSet
import numpy as np
import argparse

if __name__ == '__main__':
    prog_name = 'Softmax Neural Network Demo'
    description = 'This is a demo program for SoftmaxNN in liflib.dl.'
    
    ap = argparse.ArgumentParser(description = description, prog = prog_name)
         
    ap.add_argument('dataset', action = 'store')
    ap.add_argument('-t', '--tol', action = 'store', dest = 'tol', 
                    type = float, default = 1e-5,
                    help = 'tolerance for testing convergence')
    ap.add_argument('-b', '--batch', action = 'store', dest = 'batch_size',
                    type = int, default = 50,
                    help = 'batch size when doing mini-batch gradient descent') 
    ap.add_argument('-r', '--random', action = 'store', dest = 'random', 
                    type = bool, default = True,
                    help = 'whether randomly selecting training samples')                                
    ap.add_argument('-m', '--maxiters', action = 'store', type = int,
                    dest = 'max_iters', default = None,
                    help = 'output test result to specific path')     
    ap.add_argument('-s', '--step', action = 'store', type = float,
                    dest = 'step', default = 1.0,
                    help = 'step size of sgd')
    ap.add_argument('-a', '--anneal', action = 'store', dest = 'anneal',
                    type = int, default = 100000,
                    help = 'decrease step size every period of time') 
    ap.add_argument('-d', '--display', action = 'store', dest = 'display',
                    type = int, default = 100,
                    help = 'display information every period of time') 
    ap.add_argument('-v', '--save', action = 'store', dest = 'save',
                    type = int, default = 10000,
                    help = 'save progress to cache files every period of time') 
    ap.add_argument('-l', '--layer', action = 'append', dest = 'layer',
                    type = int,
                    help = 'add one hidden layer with supplied size') 
    args = ap.parse_args()

    dataset = DataSet.load(args.dataset) 
    print 'Data loaded! n_samples = %d, n_features = %d, n_label_classes = %d' % (dataset.n_samples, 
                                                                                  dataset.n_features, 
                                                                                  dataset.n_label_classes)
    
    layers = [dataset.n_features]
    if args.layer:
        for h in args.layer:
            layers.append(h)
    else:
        layers.append(dataset.n_features)
    layers.append(dataset.n_label_classes)
    nn = SoftmaxNN(tuple(layers))    
    print 'A %d-layer neural network with layer size %s has been constructed.' % (nn.n_layers, nn.layer_size)
    #liflib2.gradcheck_naive(lambda x: nn.gd_wrapper(x, dataset, 10, True), np.random.randn(nn.get_parameter_count()), verbose = False)
    nn.fit(dataset, 
           gd_wrapper_options={'batch_size': args.batch_size, 
                               'randomized':args.random},
           f_min_options = {'max_iters': args.max_iters, 
                            'tol': args.tol, 
                            'step_size': args.step, 
                            'anneal_every': args.anneal,
                            'print_every': args.display, 
                            'save_params_every': args.save})
    
    n_correct = 0
    for i in xrange(dataset.n_training_samples):
        features, label = dataset.get_training_sample()
        if np.argmax(nn.predict(features)) == label:
            n_correct += 1
            
    print 'For training set: total: %d, correct: %d, accuracy = %g%%' % (dataset.n_training_samples, n_correct, n_correct * 100.0 / dataset.n_training_samples)
    
    n_correct = 0
    
    for i in xrange(dataset.n_test_samples):
        features, label = dataset.get_test_sample()
        if np.argmax(nn.predict(features)) == label:
            n_correct += 1
            
    print 'For test set: total: %d, correct: %d, accuracy = %g%%' % (dataset.n_test_samples, n_correct, n_correct * 100.0 / dataset.n_test_samples)
