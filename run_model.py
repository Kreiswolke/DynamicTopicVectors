#!/usr/bin/python

import numpy as np
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
#import matplotlib.pyplot as plt
import  pickle
import sys
import argparse
import gensim
from utils import get_data, get_wordmodel, set_up_dir,  MyCorpus, write_topic_files
from sklearn.model_selection import KFold
from proc_datasets import CorpusVectorProcessorWithLabels
from run_dtv import train_dtv
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


def remove_mean(X):
    X = X - np.mean(X, axis = 0)
    X = np.array([gensim.matutils.unitvec(x) for x in X])
    return X
    
DATA_ROOT = 'data'
WORDMODEL_FILE = 'wordmodel/GloVe/glove2word2vec300d.txt'
word_model = get_wordmodel(WORDMODEL_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--savedir', type=str,
                       help='directory to store results')
    parser.add_argument('--maxiter', type=int, default=10000,
                       help='Maximal number of training epochs')
    parser.add_argument('--miniter', type=int, default=100,
                       help='Minimal number of training epochs')
    parser.add_argument('--t', type=int,
                        help='Number of topics')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='learning rate decay')
    parser.add_argument('--case', type=str,
                   help='specifies which data to use')
    parser.add_argument('--bs', type=int, help='batchsize', default=None)
    parser.add_argument('--weight', type=str, default=None, help='Dir to dict which contains weights')
    parser.add_argument('--eta', type=float, default=0.)

    args = parser.parse_args()
    model_case = args.model
    corpus_case = args.corpus
    case = args.case
    batchsize = args.bs
    res_dir = args.savedir
    set_up_dir(res_dir) 
   

    doc_dict_train, doc_dict_test, doc_dict_valid, _ = get_data(str(corpus_case), root_dir=DATA_ROOT)

    Processor = CorpusVectorProcessorWithLabels(model = word_model)


    print('Normal training: train/test split used')
    res_dict = {}
    res_dict['model'] = model_case
    res_dict['corpus'] = corpus_case
    res_dict['batchsize'] = batchsize
    res_dict['n_topics'] = args.t
    res_dict['maxiter'] = args.maxiter
    res_dict['miniter'] = args.miniter
    res_dict['lr'] = args.lr


    train_keys, test_keys = list(doc_dict_train.keys()), list(doc_dict_test.keys())   
    valid_keys = list(doc_dict_valid.keys())
    np.random.shuffle(train_keys)

    data_dict_train = {k: doc_dict_train[k] for k in train_keys}
    data_dict_test = {k: doc_dict_test[k] for k in test_keys}

    data_train, data_test,  y_train, y_test, vocab_train, res_svd_train = Processor.__call__(data_dict_train, data_dict_test, corpus_key = 'corpus', whiten=True)
    data_valid, y_valid = Processor.proc_single_doc_dict(doc_dict_valid, vocab_train, corpus_key = 'corpus', whiten=True)

    vocab_size = len(vocab_train)

    print('Training corpus: {}, Test corpus: {}, Vocabulary: {} '.format(np.shape(data_train), np.shape(data_test), vocab_size))

    if model_case in ['DTV']:

        res_dict_stv = train_dtv(args, data_train, data_test, data_valid , y_train, y_test, y_valid, vocab_train, word_model)
        res_dict.update(res_dict_stv)
        res_dict['wordmodel'] = WORDMODEL_FILE
        
        
    elif model_case in ['Baseline']:
        
        T = np.random.normal(size=(args.t,300))
        T_norm = np.array([gensim.matutils.unitvec(x) for x in T])
        res_dict['topics'] = T_norm


    
    res_dict['vocab_size'] = vocab_size
    res_dict['train_keys'] = train_keys
    res_dict['test_keys'] = test_keys
    res_dict['valid_keys'] = valid_keys
    res_dict['case'] = case
    res_str =  'res_n_{}.p'.format(res_dict['n_topics'])
    res_dict_file = os.path.join(res_dir, res_str)

    pickle.dump(res_dict, open(res_dict_file, "wb" ))

    # Write topic words
    topic_file = os.path.join(res_dir, 'topics_n_{}.txt'.format(res_dict['n_topics']))

    if model_case in ['DTV', 'Baseline']:
        res_topics = write_topic_files(res_dict, topic_file, vocab_train, remove_mean = False,  word_model_file = WORDMODEL_FILE)

    res_dict['topic_dict'] = res_topics

    pickle.dump(res_dict, open(os.path.join(res_dir, 'res_n_{}.p'.format(str(args.t))), "wb" ))
    
  

if __name__ == "__main__":
    main()
