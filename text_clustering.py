# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 18:26:40 2016

@author: lifu
"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import jieba
import os

def extract_qa_pairs(text):
    lines = text.split('\n')[:-1]
    evenLines = map(lambda p: p[1][8:], 
                    filter(lambda p: p[0] & 1 == 0, enumerate(lines)))
    oddLines = map(lambda p: p[1][6:], 
                   filter(lambda p: p[0] & 1 == 1, enumerate(lines)))
    return evenLines, oddLines

if __name__ == '__main__':    
    textPath = '/mnt/shared/ACL/test_set'
    answerClusterDir = '/mnt/shared/ACL/answerClusters'
    questionClusterDir = '/mnt/shared/ACL/questionClusters'
    n_clusters = 100
    
    with open(textPath) as f:
        text = f.read()
    questions, answers = extract_qa_pairs(text)

    tv = TfidfVectorizer()
    sparseX = tv.fit_transform(map(lambda t: ' '.join(jieba.cut(t)), answers))
    km = KMeans(n_clusters = n_clusters)
    km.fit(sparseX)
    
    clusteredAns = [[] for i in xrange(n_clusters)]
    clusteredQst = [[] for i in xrange(n_clusters)]
    
    for i in xrange(len(questions)):
        clusteredAns[km.labels_[i]].append(answers[i])
        clusteredQst[km.labels_[i]].append(questions[i])
        
    for i in xrange(n_clusters):
        print 'Generating cluster #', i, '...',
        answerClusterPath = os.path.join(answerClusterDir, str(i) + '.txt')
        questionClusterPath = os.path.join(questionClusterDir, str(i) + '.txt')
        with open(answerClusterPath, 'w') as f:
            f.write('\n'.join(clusteredAns[i]))
        with open(questionClusterPath, 'w') as f:
            f.write('\n'.join(clusteredQst[i]))
        print 'Done'
    print 'Success!'