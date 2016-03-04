# -*- coding: utf-8 -*-
"""
This is a tfidf text classifier.
@author Lifu Huang
"""

import numpy as np
import jieba
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

class Utils:
    @staticmethod
    def collect_filePaths(rootDir, mxNum = None):    
        result = []
        i = 0
        for dirpath, dirnames, filenames in os.walk(rootDir):
            for filename in filenames:
                i += 1
                if mxNum and i > mxNum:  
                    break
                result.append(os.path.join(dirpath, filename))
        return result
        
    @staticmethod
    def get_corpus(filePaths, verbose = True):   
        corpus = []
        for i in xrange(len(filePaths)):
            filePath = filePaths[i]
            if verbose:
                print 'Processing', filePath, 
            with open(filePath) as f:
                corpus.append(' '.join(jieba.cut(f.read())))
            if verbose:
                print 'Done'
        return corpus

    @staticmethod
    def short_id(path):
        return os.path.splitext(os.path.basename(path))[0]
    
    @staticmethod
    def medium_id(path):
        return os.path.basename(path)
        
    @staticmethod
    def long_id(path):
        return path
 
class TfidfClassifier:
    def __init__(self):
        self.fitted = False
        self.idf = None
        self.vocabulary = None
        self.idMapping = None
        self.tfidfMat = None
        
    def fit_sample_directory(self, sampleDir, mxNum = None,
                             idGenerator = None, verbose = True):
        if not idGenerator:
            idGenerator = Utils.short_id
        return self.fit_sample_files(Utils.collect_filePaths(sampleDir, mxNum),
                                     idGenerator, verbose)
            
    def fit_sample_files(self, filePaths, idGenerator = None, verbose = True):
        if not idGenerator:
            idGenerator = Utils.short_id
        if verbose:
            print 'Building corpus...'
        corpus = Utils.get_corpus(filePaths, verbose)
        if verbose:
            print 'Done! get', len(corpus), 'sample(s) in total.'
        
        if verbose:
            print 'Generating tf-idf matrix...' 
        tv = TfidfVectorizer()
        self.tfidfMat = tv.fit_transform(corpus)
        if verbose:
            print 'Done! Generated a', self.tfidfMat.shape, 'matrix!'

        self.idMapping = dict()   
        for i in xrange(len(filePaths)):
            self.idMapping[idGenerator(filePaths[i])] = i
        self.idf = tv.idf_
        self.vocabulary = tv.vocabulary_
        self.fitted = True
        
    def check_fitted(self):        
        if not self.fitted:
            raise Exception('Model cannot be used before fitted to a dataset.')
        
    def dump_parameters(self, target):
        self.check_fitted()
        with open(target, 'w') as f:
            pickle.dump((self.vocabulary, self.idf, 
                         self.idMapping, self.tfidfMat), f) 
    
    def load_parameters(self, target):
        with open(target, 'r') as f:
            (self.vocabulary, self.idf, 
             self.idMapping, self.tfidfMat) = pickle.load(f) 
        self.fitted = True
        
    def get_feature_count(self):
        self.check_fitted()
        return self.tfidfMat.shape[1]
        
    def get_sample_count(self):
        self.check_fitted()
        return self.tfidfMat.shape[0]
        
    def get_feature_names(self):
        self.check_fitted()
        return self.vocabulary.keys()
        
    def get_sample_ids(self):
        self.check_fitted()
        return self.idMapping.keys()
        
    def calculate_tfidf(self, text):
        self.check_fitted()
        cv = CountVectorizer(vocabulary = self.vocabulary)
        targetCount =  cv.fit_transform([' '.join(jieba.cut(text))]).toarray()
        total = float(targetCount.sum())
        if total == 0:
            return targetCount
        targetTf = targetCount / total
        unnormalizedTfidf = np.multiply(targetTf, self.idf)
        return unnormalizedTfidf / np.linalg.norm(unnormalizedTfidf, ord=2)

    def get_sample_vector(self, sampleId):
        self.check_fitted()
        return self.tfidfMat[self.idMapping[sampleId], :].toarray()
    
    def calculate_similarity(self, text, sampleId):
        self.check_fitted()
        targetTfidf = self.calculate_tfidf(text)
        sampleTfidf = self.get_sample_vector(sampleId)
        return float(np.dot(targetTfidf, sampleTfidf.T))
    
    def get_most_similar_samples(self, text, k = 1):
        self.check_fitted()
        product = np.dot(self.tfidfMat.toarray(), self.calculate_tfidf(text).T)
        return sorted(self.get_sample_ids(),
                      key = (lambda id: product[self.idMapping[id]]), 
                      reverse = True)[:k]

if __name__ == '__main__':  
    options = ['test']
    textFilePath = '/mnt/shared/ACL/Q.txt'
    parameterPath = '/mnt/shared/ACL/saves'
    questionPath = '/mnt/shared/ACL/questions'
    
    def check_tfidf_accuracy(qstDir, verbose = True):
        print 'Collecting question filenames...'
        qstFilePaths = TfidfClassifier.collect_filePaths(qstDir)
        print 'Done! Collected', len(qstFilePaths), 'file path(s) in total.'
        
        cnt = 0
        c = TfidfClassifier()
        c.load_parameters(parameterPath)
        
        for i in xrange(len(qstFilePaths)):
            print 'checking', qstFilePaths[i], 
            with open(qstFilePaths[i]) as f:
                targetTfidf = c.calculate_tfidf(f.read())        
            originalTfidf = c.get_sample_vector(c.short_id(qstFilePaths[i]))
            print 'error:', np.linalg.norm(targetTfidf -originalTfidf, ord = 1),
            if np.sum(np.abs(targetTfidf - originalTfidf) > 
                    (targetTfidf + originalTfidf) * 0.05) == 0:
                cnt += 1
        print 'Final result:', cnt * 100/ len(qstFilePaths), '% are correct!'

    for option in options:
        if option == 'dump':   
            c = TfidfClassifier()
            c.fit_sample_directory(questionPath)
            c.dump_parameters(parameterPath) 
        elif option == 'check':
            check_tfidf_accuracy(qstDir = '/mnt/shared/ACL/questions/') 
        elif option == 'test':
            c = TfidfClassifier()
            c.load_parameters(parameterPath)
            with open(textFilePath) as f:
                for i, line in enumerate(f):
                    result = c.get_most_similar_samples(line, k = 5)
                    for item in result:
                        print item,
                    print 
