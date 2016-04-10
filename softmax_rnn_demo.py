# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 17:49:25 2016

@author: lifu
"""
from liflib2.dl import RNNLM

filename = '/home/lifu/text'

model = RNNLM(filename, (50, ))
model.start_training()