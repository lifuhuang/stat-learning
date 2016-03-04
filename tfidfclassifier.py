# -*- coding: utf-8 -*-
"""

This temporary script file is located here:
/home/lifu/.spyder2/.temp.py
"""

import numpy as np
import scipy as sp
import scipy.io
import jieba
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

class TfidfClassifier:
    def TfidfClassifier(self):
        self.fitted = False
        self.idf = None
        self.sampleVecs = None
        
    def collect_filenames(self, rootDir, maxNum = None):    
        result = []
        i = 0
        for dirpath, dirnames, filenames in os.walk(rootDir):
            for filename in filenames:
                i += 1
                if maxNum and i > maxNum:  
                    break
                result.append(filename)
        return result
        
    def get_corpus(self, rootDir, filenames, verbose = True):   
        corpus = []
        for i in xrange(len(filenames)):
            filePath = os.path.join(rootDir, filenames[i])
            if verbose:
                print 'Processing', filePath, 
            with open(filePath) as f:
                corpus.append(' '.join(jieba.cut(f.read())))
            if verbose:
                print 'Done'
        return corpus
        
    def fit_sample_files(self, rootDir, verbose = True):
        if verbose:
            print 'Collecting sample filenames...'
        filenames = self.collect_filenames(rootDir, 100)
        if verbose:
            print 'Done! Collected', len(filenames), 'filename(s) in total.'
            
        if verbose:
            print 'Building corpus...'
        corpus = self.get_corpus(rootDir, filenames, verbose)
        if verbose:
            print 'Done! get', len(corpus), 'sample(s) in total.'
        
        if verbose:
            print 'Generating tf-idf matrix...'  
        tv = TfidfVectorizer()
        tfidfMat = tv.fit_transform(corpus)
        if verbose:
            print 'Done! Generated a', tfidfMat.shape, 'matrix!'

        self.sampleVecs = dict()        
        for i in xrange(len(filenames)):
            self.sampleVecs[filenames[i]] = tfidfMat[i, :]
        self.idf = tv.idf_
        self.featureNames = tv.get_feature_names()
        self.fitted = True
        
    def dump_parameters(self, target):
        if not self.fitted:
            raise Exception('Model cannot be used before fitted to a dataset.')
        with open(target, 'w') as f:
            pickle.dump((self.featureNames, self.idf, self.sampleVecs), f) 
    
    def load_parameters(self, target):
        with open(target, 'r') as f:
            self.featureNames, self.idf, self.sampleVecs = pickle.load(f) 
        self.fitted = True
        
    def calculate_tfidf(self, text):
        if not self.fitted:
            raise Exception('Model cannot be used before fitted to a dataset.')
        cv = CountVectorizer(vocabulary = self.featureNames)
        targetText =  cv.fit_transform([' '.join(jieba.cut(text))])
        targetTf = (targetText / targetText.sum()).toarray()
        unnormalizedTfidf = np.multiply(targetTf, self.idf)
        return unnormalizedTfidf / np.linalg.norm(unnormalizedTfidf, ord=2)
        
    def calculate_similarity(self, text, sampleId):
        targetTfidf = self.calculate_tfidf(text)
        return float(np.dot(targetTfidf, self.sampleVecs[sampleId].toarray().T))
    
    def find_most_similar_samples(self, text, k = 1):
        return sorted((i for i in self.sampleVecs), 
                      key = lambda i: self.calculate_similarity(text, i), 
                        reverse = True)[:k]
 
if __name__ == '__main__': 
    c = TfidfClassifier()
    c.load_parameters('/mnt/shared/ACL/saves')
    
