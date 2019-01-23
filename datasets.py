import gensim
import numpy as np
import os
import itertools
import xxhash
from gensim import utils, matutils
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL
import uuid
import pickle
import scipy
from nltk.corpus import stopwords
from nltk import word_tokenize
import collections
from gensim.parsing.preprocessing import preprocess_string, \
strip_multiple_whitespaces,strip_non_alphanum, strip_numeric, \
strip_punctuation, strip_tags, strip_short
from utils import download_file, set_up_dir, get_wordmodel, get_data
import subprocess
from nltk.corpus import stopwords
import pandas
import re

stop_words = stopwords.words('english')
stop_words = stop_words 

class TwentyNewsGroup(object):
    def __init__(self, root_dir):
        self.name = '20NewsGroupCachopo'
        self.txt_source_train = os.path.join(root_dir, self.name,  '20ng-train-all-terms.txt')
        self.txt_source_test = os.path.join(root_dir, self.name, '20ng-test-all-terms.txt')
        self.root_dir = root_dir


    def download(self):
        url_test = 'http://ana.cachopo.org/datasets-for-single-label-text-categorization/20ng-test-all-terms.txt?attredirects=0&d=1'
        url_train = 'http://ana.cachopo.org/datasets-for-single-label-text-categorization/20ng-train-all-terms.txt?attredirects=0&d=1'
        
        set_up_dir(os.path.join(self.root_dir, self.name))
      
        if not os.path.isfile(self.txt_source_test):
             download_file(url_test, self.txt_source_test)
        
        if not os.path.isfile(self.txt_source_train):
            download_file(url_train, self.txt_source_train)

    def get_train(self):
        for line in open(self.txt_source_train, 'r').readlines():
            _id = str(uuid.uuid4())
            split = line.split('\t') 
            label, text = split[0], split[1]
            words = text.strip().split()
            
            words = [w for w in words if w not in stop_words]
            if len(words) > 0:            
                yield gensim.models.doc2vec.TaggedDocument(words = words, tags = ['_'.join(['train', label, _id])])
            
    def get_test(self):
        for line in open(self.txt_source_test, 'r').readlines():
            _id = str(uuid.uuid4())
            split = line.split('\t') 
            label, text = split[0], split[1]
            words = text.strip().split()
            
            words = [w for w in words if w not in stop_words]
            if len(words) > 0:            
                yield gensim.models.doc2vec.TaggedDocument(words = words, tags = ['_'.join(['test', label, _id])])


class RCV1(object):
    def __init__(self, root_dir):
        # Swapping test and train
        self.name = 'RCV1'
        self.txt_source_train = [os.path.join(root_dir, self.name, 'lyrl2004_tokens_test_pt{}.dat.gz'.format(i)) for i in range(4)]
        self.txt_source_test = os.path.join(root_dir, self.name, 'lyrl2004_tokens_train.dat.gz')
        self.root_dir = root_dir

    def download(self):
        url1 = ['http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt{}.dat.gz'.format(i) for i in range(4)]   
        url2 = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz'   
      
        set_up_dir(os.path.join(self.root_dir, self.name))
 
        for x, url in zip(self.txt_source_train, url1):
            if not os.path.isfile(x.replace('.gz','')):
                download_file(url, x)
                p = subprocess.Popen(['gunzip', x ], cwd=os.path.join(self.root_dir, self.name))
                p.wait()

        if not os.path.isfile(self.txt_source_test.replace('.gz','')):
            download_file(url2, self.txt_source_test)               
            p = subprocess.Popen(['gunzip', self.txt_source_test ], cwd=os.path.join(self.root_dir, self.name))
            p.wait()

    def get_train(self):
        for txt_source in self.txt_source_train:
            for line in open(txt_source.replace('.gz',''), 'r').readlines():
                #Begin text
                if line.startswith( '.I' ):
                    raw = []

                raw.append(line)
                
                #End text
                if line == '\n':
                    _id = str(uuid.uuid4())
                    label = raw[0].strip().replace(' ', '_')
                    text = ' '.join(raw[2:]).strip().split()
                    words = [w for w in text if w not in stop_words]
                    raw = []
                    yield gensim.models.doc2vec.TaggedDocument(words = words, tags = ['_'.join(['train', label, _id])])


            
            
    def get_test(self):
        for line in open(self.txt_source_test.replace('.gz',''), 'r').readlines():
            #Begin text
            print(self.txt_source_test)
            if line.startswith( '.I' ):
                raw = []

            raw.append(line)

            #End text
            if line == '\n':
                _id = str(uuid.uuid4())
                label = raw[0].strip().replace(' ', '_')
                text = ' '.join(raw[2:]).strip().split()
                words = [w for w in text if w not in stop_words]
                raw = []
                yield gensim.models.doc2vec.TaggedDocument(words = words, tags = ['_'.join(['test', label, _id])])

        
        

        
        
        
        
        
        
        
        
        
