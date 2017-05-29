# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 18:00:20 2016

@author: lifu
"""

from liflib2 import DataSet
import argparse

if __name__ == '__main__':
    prog_name = 'DataSet Demo'
    description = 'This is a demo program for DataSet in liflib.'
    
    ap = argparse.ArgumentParser(description = description, prog = prog_name)
         
    ap.add_argument('path_features', action = 'store')
    ap.add_argument('path_labels', action = 'store')
    ap.add_argument('-o', '--output', action = 'store', dest = 'output_path',
                    default = 'dataset.dat')
    args = ap.parse_args()
    
    ds = DataSet.load_from_txt_files(args.path_features, args.path_labels)
    ds.dump(args.output_path)
    
    print 'Success! Data has been dumped to %s' % args.output_path