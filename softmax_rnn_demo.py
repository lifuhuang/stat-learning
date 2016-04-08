# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 17:49:25 2016

@author: lifu
"""
import jieba
import numpy as np
from liflib2.dl import SoftmaxRNN

filename = '/home/lifu/text'

def padding(sentence):
    return [u'<s>'] + sentence + ['</s>']

sentences = []
word_to_num = {u'<s>':0, u'</s>':1}

with open(filename, 'r') as fp:
    for line in fp:
        raw_sentence = list(jieba.cut(line))
        for word in raw_sentence:
            if word not in word_to_num:
                word_to_num[word] = len(word_to_num)
        sentences.append(padding(raw_sentence))
num_to_word = {word_to_num[word]: word for word in word_to_num}

model = SoftmaxRNN((len(word_to_num), 50, 50, len(word_to_num)))

print 'Collected %d sentences in total.' % len(sentences)
seqs = map(lambda s: np.array(map(word_to_num.__getitem__, s)), sentences)
X = np.array(map(lambda seq:seq[:-1], seqs), dtype=object)
Y = np.array(map(lambda seq:seq[1:], seqs), dtype=object)
model.train_sgd(X, Y, idxiter=model.randomiter(100000, len(X), 1), printevery=100, costevery=1000)

for i in xrange(5):
    st, loss = model.generate_sequence(word_to_num[u'<s>'], word_to_num[u'</s>'])
    print ''.join(map(num_to_word.__getitem__, st)), loss
    
