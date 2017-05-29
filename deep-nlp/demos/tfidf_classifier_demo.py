# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 11:24:37 2016

@author: lifu
"""

import argparse
import numpy as np
from liflib2.text import Utils
from liflib2.text import TfidfClassifier

def check_tfidf_accuracy(classifier, sample_dir, verbose = True):
    if verbose:
        print 'Collecting sample filenames...'
        sample_file_paths = Utils.collect_file_paths(sample_dir)
    if verbose:
        print 'Done! Collected', len(sample_file_paths), 'file path(s) in total.'
        
    cnt = 0
    
    for i in xrange(len(sample_file_paths)):
        if verbose:
            print 'checking', sample_file_paths[i], 
        with open(sample_file_paths[i]) as f:
            targetTfidf = classifier.calculate_tfidf(f.read())        
        originalTfidf = classifier.get_sample_vector(
            Utils.short_id(sample_file_paths[i]))
        if verbose:
            print 'error:', np.linalg.norm(targetTfidf -originalTfidf, ord = 1)
        if np.sum(np.abs(targetTfidf - originalTfidf) > 
                (targetTfidf + originalTfidf) * 0.01) == 0:
            cnt += 1
    print 'Final result:', cnt * 100/ len(sample_file_paths), '% are correct!'

if __name__ == '__main__':      
    prog_name = 'TfidfClassifier Demo'
    description = 'This is a demo program for TfidfClassifier in liflib.text.'
    ap = argparse.ArgumentParser(description = description, prog = prog_name)
    
    fit_or_load = ap.add_mutually_exclusive_group(required = True)    
    fit_or_load.add_argument('-f', '--fit', action = 'store',
                             dest = 'training_set_dir', 
                             help = 'train model with given files')
    fit_or_load.add_argument('-l', '--load', action = 'store',
                             dest = 'param_path', 
                             help = 'load parameter from specific file path')        
    ap.add_argument('-d', '--dump', action = 'store', dest = 'dump_path', 
                    help = 'dump fitted parameters to specific path')
    ap.add_argument('-c', '--check', action = 'store', dest = 'check_dir',
                    help = 'check accuracy of calculated tfidf')
    ap.add_argument('--test', '-t', action = 'store', dest = 'test_file_path',
                    help = 'test data file')                                 
    ap.add_argument('-o', '--output', action = 'store', 
                    dest = 'test_result_path', 
                    help = 'output test result to specific path')     
    ap.add_argument('-v', '--verbose', action = 'store_true', dest = 'verbose',
                    help = 'use verbose mode')  
    ap.set_defaults(verbose = True, test_result_path = 'test_result')
    args = ap.parse_args()

 
    c = TfidfClassifier()
    # ways of getting parameters right
    if args.training_set_dir:  
        c.fit_sample_directory(args.training_set_dir, verbose = args.verbose)
        # dump fitted parameters to file        
        if args.dump_path:
            c.dump_parameters(args.dump_path)           
    elif args.param_path:
        c.load_parameters(args.param_path)
    
    # test
    if args.test_file_path:
        allResult = []
        with open(args.test_file_path) as f:
            for i, line in enumerate(f):
                if i & 1:
                    continue
                question = line.decode('utf-8')[8:]
                if args.verbose:                
                    print question
                allResult.append(
                    ' '.join(c.get_most_similar_samples(question, k = 5)))
        with open(args.test_result_path, 'w') as f:
            f.write('\n'.join(allResult))

    #check 
    if args.check_dir:
        check_tfidf_accuracy(classifier = c, sample_dir = args.check_dir,
                             verbose = args.verbose)   
