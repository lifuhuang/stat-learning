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