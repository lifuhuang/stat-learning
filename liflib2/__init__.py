# -*- coding: utf-8 -*-
"""
This is the initialization script of package liflib.

@author Lifu Huang
"""

import numpy as np
import random
import cPickle as pickle
import glob
import os.path as op

def softmax(x):
    """ 
    Calculates the Softmax value of the array-like input.
    """
    if x.ndim == 1:
        x_exp = np.exp(x - np.max(x))
        return x_exp / np.sum(x_exp)
    elif x.ndim == 2:
        x_exp = np.exp(x - np.max(x, axis = 1).reshape(-1, 1))
        return x_exp / np.sum(x_exp, axis = 1).reshape(-1, 1)
    else:
        raise Exception("Wrong dimension!")

def sigmoid(x):
    """
    Calculates the Sigmoid value of the array-like input. 
    """
    return 1.0 / (1.0 + np.exp(-x))

def gradcheck_naive(f, x, verbose = False):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost 
        and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        if verbose:
            print 'Checking %s...' % ix,
            
        x[ix] += h
        random.setstate(rndstate)  
        fxp = f(x)[0]
        
        x[ix] -= 2 * h
        random.setstate(rndstate)  
        fxn = f(x)[0]
        
        x[ix] += h
        numgrad = (fxp - fxn) / (2 * h)
        
        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], 
                                                                   numgrad)
            return
    
        it.iternext() # Step to next dimension
        if verbose:
            print 'pass'
    print "Gradient check passed!"    

def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets 
    iteration start 
    """
    st, params, state = 0, None, None
    for f in glob.glob("saved_params_*.npy"):
        st = max(st, int(op.splitext(op.basename(f))[0].split("_")[2]))
            
    if st > 0:
        with open("saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
    return st, params, state
    
def save_params(iter, params):
    with open("saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

def sgd(f, x0, tol = 1e-5, step_size = 0.3, 
        max_iters = None, postprocessing = None, 
        print_every = None, anneal_every = None, save_params_every = None):
    """ 
    Stochastic Gradient Descent 
    """

    if save_params_every:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            if anneal_every:
                step_size *= 0.5 ** (start_iter // anneal_every)
            random.setstate(state)
    else:
        start_iter = 0
    
    x = x0
    
    if not postprocessing:
        postprocessing = lambda x: x
    
    old_cost = float('inf')
    it = start_iter + 1
    while not max_iters or it <= max_iters:
        cost, grad = f(x)
        if abs(cost - old_cost) < tol:
            break
        x = postprocessing(x - step_size * grad)
        if print_every and it % print_every == 0:
            print 'Iterated %d times, cost = %g' % (it, cost)
        if save_params_every and it % save_params_every == 0:
            save_params(it, x)
        if anneal_every and it % anneal_every == 0:
            step_size *= 0.5
        it += 1
    
    return x


class DataSet:
    def __init__(self, ratio = (0.8, 0, 0.2)):
        """
        Initializes a Dataset. 
        """        
        self.ratio = ratio
        if sum(self.ratio) != 1:
            raise ValueError("tuple ratio must sum to 1")
        
        self.training_cnt = None
        self.cv_cnt = None
        self.test_cnt = None
        
        self.n_features = None
        self.n_samples = None
        self.n_label_classes = None
        
        self.n_training_samples = None
        self.n_test_samples = None
        self.n_cv_samples = None
        
        self.training_features = None
        self.training_labels = None        
        self.cv_features = None
        self.cv_labels = None
        self.test_features = None
        self.test_labels = None
        self.filled = False
        
        
    def load_features_labels(self, features, labels):
        """
        Loads data to DataSet.
        """
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Inconcistent sample number!")
        self.n_samples, self.n_features = features.shape
        self.n_label_classes = np.unique(labels).size
        
        self.n_training_samples = int(self.n_samples * self.ratio[0])
        self.n_cv_samples = int(self.n_samples * self.ratio[1])
        self.n_test_samples = int(self.n_samples - self.n_training_samples 
                                                 - self.n_cv_samples)
        
        shuffled_ids = [i for i in xrange(self.n_samples)]
        random.shuffle(shuffled_ids)
        
        self.training_features = features[shuffled_ids[:
            self.n_training_samples]]
        self.training_labels = labels[shuffled_ids[:self.n_training_samples]]
        
        self.cv_features = features[shuffled_ids[self.n_training_samples:
            self.n_training_samples+self.n_cv_samples]]
        self.cv_labels = labels[shuffled_ids[self.n_training_samples:
            self.n_training_samples+self.n_cv_samples]]
        
        self.test_features = features[shuffled_ids[-self.n_test_samples:]]
        self.test_labels = labels[shuffled_ids[-self.n_test_samples:]]
        
        self.reset()
        self.filled = True        
   
    def dump(self, path):
        with open(path, 'w') as f:
            pickle.dump(self, f)
        
    @staticmethod
    def load_from_txt_files(feature_file_path, label_file_path):
        """
        Loads data from files.
        """
        dataset = DataSet()
        dataset.load_features_labels(np.loadtxt(feature_file_path), 
                                     np.loadtxt(label_file_path))
        return dataset
 
    @staticmethod    
    def load(path):        
        with open(path, 'r') as f:
            ds = pickle.load(f)
        return ds
        
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
            
