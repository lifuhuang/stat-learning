# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:04:00 2016

@author: lifu
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import jieba
import argparse
import os

def extract_qa_pairs(text):
    lines = text.split('\n')[:-1]
    evenLines = map(lambda p: p[1][8:], 
                    filter(lambda p: p[0] & 1 == 0, enumerate(lines)))
    oddLines = map(lambda p: p[1][6:], 
                   filter(lambda p: p[0] & 1 == 1, enumerate(lines)))
    return evenLines, oddLines

if __name__ == '__main__':    
    ap = argparse.ArgumentParser()
    ap.add_argument(dest = 'input_path', 
                    help = 'file containing text to be clustered')
    ap.add_argument(dest = 'output_dir',
                    help = 'directory of the clustered result')
    ap.add_argument(dest = 'n_qst_clusters', type = int,
                    help = 'number of question clusters')
    ap.add_argument(dest = 'n_ans_clusters', type = int,
                    help = 'number of answer clusters')
    args = ap.parse_args()
    
    with open(args.input_path) as f:
        text = f.read()
    questions, answers = extract_qa_pairs(text)

    print 'Clustering questions...'
    tv_qst = TfidfVectorizer()
    X_qst = tv_qst.fit_transform(map(lambda t: ' '.join(jieba.cut(t)), questions))
    km_qst = KMeans(n_clusters = args.n_qst_clusters)
    km_qst.fit(X_qst)      
    clustered_qst = [[] for i in xrange(args.n_qst_clusters)]    
    for i in xrange(len(questions)):
        clustered_qst[km_qst.labels_[i]].append(questions[i])
    question_cluster_dir = os.path.join(args.output_dir, 'question_clusters/')
    if not os.path.exists(question_cluster_dir):
        os.makedirs(question_cluster_dir)
    for i in xrange(args.n_qst_clusters):
        print 'Generating question cluster #%d..' % i,
        question_cluster_path = os.path.join(question_cluster_dir,
                                             str(i) + '.txt')
        with open(question_cluster_path, 'w') as f:
            f.write('\n'.join(clustered_qst[i]))
        print 'Done'
    print 'All %d question clusters have been generated!' % args.n_qst_clusters
    
    
    print 'Clustering answers...'  
    tv_ans = TfidfVectorizer()
    X_ans = tv_ans.fit_transform(map(lambda t: ' '.join(jieba.cut(t)), answers))
    km_ans = KMeans(n_clusters = args.n_ans_clusters)
    km_ans.fit(X_ans)  
    clustered_ans = [[] for i in xrange(args.n_ans_clusters)]
    for i in xrange(len(answers)):
        clustered_ans[km_ans.labels_[i]].append(i)
    answer_cluster_dir = os.path.join(args.output_dir, 'answer_clusters/')
    if not os.path.exists(answer_cluster_dir):
        os.makedirs(answer_cluster_dir)
    for i in xrange(args.n_ans_clusters):
        print 'Generating answer cluster #%d..' % i,
        answer_cluster_path = os.path.join(answer_cluster_dir,
                                             str(i) + '.txt')
        with open(answer_cluster_path, 'w') as f:
            f.write('\n'.join(clustered_qst[i]))
        print 'Done!'
    print 'Done!'
    print 'All %d answer clusters have been generated!' % args.n_ans_clusters

    with open(os.path.join(args.output_dir, 'questions.txt'), 'w') as f:
        f.write('\n'.join(questions))

    with open(os.path.join(args.output_dir, 'answers.txt'), 'w') as f:
        f.write('\n'.join(answers))
        
    print 'Success!'
