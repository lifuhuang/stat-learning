# -*- coding: utf-8 -*-
"""
This is the text processing part of liflib.

@author Lifu Huang
"""

import numpy as np
import jieba
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

class Utils:
    @staticmethod
    def collect_file_paths(root_dir, max_num = None):  
        result = []
        i = 0
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                i += 1
                if max_num and i > max_num:  
                    break
                
                result.append(os.path.join(dirpath, filename))
                
        return result
        
    @staticmethod
    def short_id(path):
        return os.path.splitext(os.path.basename(path))[0]
    
    @staticmethod
    def medium_id(path):
        return os.path.basename(path)
        
    @staticmethod
    def long_id(path):
        return path
 
def jieba_tokenizer(sent):
    return jieba.cut(sent)
 
class TfidfClassifier:
    def __init__(self):
        self.fitted = False
        self.id_mapping = None
        self.tfidf_matrix = None
        
    def fit_sample_directory(self, sample_dir, max_num = None,
                             id_generator = None, verbose = True):
        if not id_generator:
            id_generator = Utils.short_id
            
        return self.fit_sample_files(Utils.collect_file_paths(sample_dir, max_num),
                                     id_generator, verbose)
            
    def fit_sample_files(self, file_paths, id_generator = None, verbose = True):
        if not id_generator:
            id_generator = Utils.short_id
            
        if verbose:
            print 'Building corpus...'
        
        corpus = []
        for file_path in file_paths:
            with open(file_path, 'r') as fp:
                 corpus.append(fp.read())
            
        if verbose:
            print 'Done! get', len(corpus), 'sample(s) in total.'
        
        if verbose:
            print 'Generating tf-idf matrix...' 
            
        tv = TfidfVectorizer(tokenizer=jieba_tokenizer)
        self.tfidf_matrix = tv.fit_transform(corpus)
        self.vectorizer = tv
        if verbose:
            print 'Done! Generated a', self.tfidf_matrix.shape, 'matrix!'

        self.id_mapping = dict()   
        for i in xrange(len(file_paths)):
            self.id_mapping[id_generator(file_paths[i])] = i
            
        self.fitted = True
        
    def check_fitted(self):        
        if not self.fitted:
            raise Exception('Model cannot be used before fitted to a dataset.')
        
    def dump_parameters(self, target):
        self.check_fitted()
        with open(target, 'w') as f:
            pickle.dump((self.vectorizer, self.id_mapping, self.tfidf_matrix), f) 
    
    def load_parameters(self, target):
        with open(target, 'r') as f:
            (self.vectorizer, self.id_mapping, self.tfidf_matrix) = pickle.load(f) 
            
        self.fitted = True
        
    def get_feature_count(self):
        self.check_fitted()
        return self.tfidf_matrix.shape[1]
        
    def get_sample_count(self):
        self.check_fitted()
        return self.tfidf_matrix.shape[0]
        
    def get_feature_names(self):
        self.check_fitted()
        return self.vectorizer.get_feature_names()
        
    def get_sample_ids(self):
        self.check_fitted()
        return self.id_mapping.keys()
        
    def calculate_tfidf(self, text):
        self.check_fitted()
        return self.vectorizer.transform([text]).toarray()
        
    def get_sample_vector(self, sample_id):
        self.check_fitted()
        return self.tfidf_matrix[self.id_mapping[sample_id], :].toarray()
    
    def calculate_similarity(self, text, sample_id):
        self.check_fitted()
        target_tfidf = self.calculate_tfidf(text)
        sample_tfidf = self.get_sample_vector(sample_id)
        return float(np.dot(target_tfidf, sample_tfidf.T))
    
    def get_most_similar_samples(self, text, k = 1):
        self.check_fitted()
        product = np.dot(self.tfidf_matrix.toarray(), self.calculate_tfidf(text).T)
        return sorted(self.get_sample_ids(),
                      key = (lambda id: product[self.id_mapping[id]]), 
                      reverse = True)[:k]
                      
