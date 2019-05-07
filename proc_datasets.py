import gensim
import numpy as np
import os
import itertools
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
from utils import download_file, set_up_dir, get_wordmodel, get_data, MyCorpus
dict_years = {i:k for i,k in zip(range(1991,2017), range(26))} #for NYT data
dict_months = {i:k for i,k in zip(range(1,13), range(12))} #for NYT data

def svd_whiten(X, return_res =False, nlargest=None):

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    if nlargest is not None:
        print('Whitening using {} eigenvectors'.format(nlargest))
        X_white = np.dot(U[:,:nlargest], Vt[:nlargest,:])
    else:
        print('Whitening using all eigenvectors')

        X_white = np.dot(U, Vt)
        
        
    X_white_norm = np.array([matutils.unitvec(x) for x in X_white])
    
    if return_res == False:
        return X_white_norm
    else:
        res_svd = {'U': U, 's':s, 'Vt': Vt, 'nlargest': nlargest}
        return X_white_norm, res_svd



def get_doc_hash(doc):
    import xxhash
    subgraph_id = xxhash.xxh64()
    subgraph_id.update(''.join([str(graph_date.date()).replace('-', '')] + vertices))
    return str(vertices_hash.intdigest()), str(simhash.compute(hashed_ids)), str(subgraph_id.intdigest())

def get_training_and_testing_sets(file_list, split = 0.8):
    split_index = int(np.floor(len(file_list) * split))
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

def split_doc_dict(doc_dict, split_ratio):
    keys = list(doc_dict.keys())
    np.random.seed(42)
    np.random.shuffle(keys)
    training_keys, testing_keys = get_training_and_testing_sets(keys, split_ratio)

    training_doc_dict = collections.OrderedDict((k, doc_dict[k]) for k in training_keys)
    testing_doc_dict = collections.OrderedDict((k, doc_dict[k]) for k in testing_keys)

    assert len(training_doc_dict)+len(testing_doc_dict) == len(doc_dict)
    return training_doc_dict, testing_doc_dict


def reduce_doc_dict(doc_dict, n_samples):
    keys = list(doc_dict.keys())
    np.random.shuffle(keys)
    keys_keep = keys[:n_samples]
    doc_dict_reduced = collections.OrderedDict((k, doc_dict[k]) for k in keys_keep)
    assert len(doc_dict) == n_samples
    return doc_dict_reduced

class CreateItem2Vector(object):
    '''
    Class to compute vectorized embeddings from item_lists using a model

    model: word_model which assign vectors to word tokens: vec = model['cat']
    type: 'word2vec' or 'doc2vec'
    '''
    def __init__(self, model,type='word2vec'):
        self.model = model
        self.type = type

    def collect_weights(self, item, size):
        # item: word lists, e.g. ['I', 'am', 'here']

        if self.type == 'word2vec':
            vec = []
            skip_list = []
            word_list = [word for word in item if word in self.model.vocab]
            [vec.append(self.model[word]) if word in self.model.vocab else skip_list.append(word) for word in item]
            
            if len(vec) > 1:
                doc_vec = matutils.unitvec(np.array(vec).mean(axis=0)).astype(REAL)
            else:
                doc_vec = np.zeros(size)                

            if np.isnan(doc_vec).all():
                doc_vec = np.zeros(size)                
        elif self.type == 'doc2vec':
            doc_vec = matutils.unitvec(np.array(self.model.docvecs[item['_id']]).astype(REAL))
            skip_list = []
        else:
            raise
        return doc_vec, skip_list, word_list, item



class CorpusProcessor(object):
    def __init__(self, corpus_object, model_dir, target_dir):
        self.model_dir = model_dir
        self.target_dir = target_dir
        self.corpus = corpus_object
        self.case = self.corpus.name
    
    def __call__(self):
        word_model = get_wordmodel(self.model_dir)
        self.vectorizer = CreateItem2Vector(word_model, 'word2vec')

        info_file = open(os.path.join(self.target_dir, 'info.log'), 'a')
        

        for sentences, stage in [(self.corpus.get_test(), 'test'), (self.corpus.get_train(), 'train')]:
            doc_dict =  self.create_doc_dict(sentences)
          #  corpus_dict = self.compute_doc_mat(doc_dict)
          
            info_file.write('{}: {}\n'.format(stage, len(doc_dict)))
            pickle.dump(doc_dict, open( os.path.join(self.target_dir, "{}_doc_dict.p".format(stage)), "wb" ))
         #   pickle.dump(corpus_dict, open( os.path.join(self.target_dir, "{}_corpus_dict.p".format(stage)), "wb" ))
   
    def split(self):
        doc_dict = pickle.load(open( os.path.join(self.target_dir, "test_doc_dict.p"), "rb" ))
        doc_dict_test, doc_dict_valid = split_doc_dict(doc_dict, 0.8)

        print('Split doc_dict {} into test {} and validation {} sets'.format(len(doc_dict), len(doc_dict_test), len(doc_dict_valid)))
        pickle.dump(doc_dict_test, open( os.path.join(self.target_dir, "testsplit_doc_dict.p"), "wb" ))
        pickle.dump(doc_dict_valid, open( os.path.join(self.target_dir, "valid_doc_dict.p"), "wb" ))

    def reduce(self, doc_dict_name, n_samples):

        doc_dict = pickle.load(open( os.path.join(self.target_dir, doc_dict_name), "rb" ))
        doc_dict_reduced = reduce_doc_dict(doc_dict, n_samples)
            
        dir_ = os.path.join(self.target_dir, 'reduced')
        set_up_dir(dir_)
        pickle.dump(doc_dict_test, open( os.path.join(dir_, doc_dict_name), "wb" ))


    def create_doc_dict(self, sentences):
        doc_dict = collections.OrderedDict()
        for s in sentences:
            vec, skip, words, line = self.vectorizer.collect_weights(s.words, 300)
            doc_dict[s.tags[0]] = {'doc2vec': vec, 'skip_words': skip, 'words': words, 'raw_line': line}
        return doc_dict

    def compute_doc_mat(self, doc_dict):
        doc_mat = np.array([d['doc2vec'] for _,d in doc_dict.items()])
        N = len(doc_mat)
        corpus_raw = doc_mat[:N]
        corpus_mean = np.mean(doc_mat[:N], axis=0)[np.newaxis,:]
        corpus = corpus_raw - corpus_mean
        corpus = np.array([matutils.unitvec(row) for row in corpus])

        corpus_dict = {}
        corpus_dict['raw_corpus'] = corpus_raw
        corpus_dict['corpus'] = corpus
        corpus_dict['mean'] = corpus_mean
        return corpus_dict


class CorpusVectorProcessorWithLabels(CreateItem2Vector):
    '''
    Processing of corpus data
    '''

    def __init__(self, model=None):
        super(CorpusVectorProcessorWithLabels, self).__init__(model)

    def __call__(self, doc_dict_train, doc_dict_test, corpus_key = 'corpus', whiten = True):
        corpus_dict_train, corpus_dict_test = self.compute_doc_mat(doc_dict_train), self.compute_doc_mat(doc_dict_test)
        y_train, y_test = self.get_labels(doc_dict_train), self.get_labels(doc_dict_test)
        
        mycorpus = MyCorpus(doc_dict_train)   
        vocab_inv = {str(v):k for k,v in mycorpus.dictionary.items()}    

        if whiten == True:
            print('Whiten On')
            train_mat, res_svd_train = svd_whiten(corpus_dict_train[corpus_key], return_res = True, nlargest=None )
            test_mat, _ = svd_whiten(corpus_dict_test[corpus_key], return_res = True,  nlargest=None)        
        else:          
            print('Whiten Off')
            res_svd_train = None
            train_mat = corpus_dict_train[corpus_key]
            test_mat = corpus_dict_test[corpus_key]
        
        return train_mat, test_mat, y_train, y_test, vocab_inv, res_svd_train
    
    def proc_single_doc_dict(self, doc_dict, vocabulary, corpus_key = 'corpus', whiten = True):
        corpus_dict = self.compute_doc_mat(doc_dict)
        y = self.get_labels(doc_dict)
        
        if whiten == True:
            print('Whiten On')
            mat = svd_whiten(corpus_dict[corpus_key], return_res = False, nlargest=None )
        else:
            print('Whiten Off')
            mat = corpus_dict[corpus_key]
        return mat, y
    
    def get_labels(self, doc_dict):
        y_list = [0 for k in doc_dict.keys()]
        return y_list
         

    def create_doc_dict(self, input_dict):

        doc_dict = collections.OrderedDict()
        for k,v in input_dict.items():
            vec, skip, words, line = self.collect_weights(v['words'], 300)
            doc_dict[k] = {'doc2vec': vec, 'skip_words': skip, 'words': words, 'raw_line': line}
            assert sorted(line) == sorted(v['words'])
        return doc_dict
        

    def compute_doc_mat(self, doc_dict):
        doc_mat = np.array([d['doc2vec'] for _,d in doc_dict.items()])
        N = len(doc_mat)
        corpus_raw = doc_mat[:N]
        corpus_mean = np.mean(doc_mat[:N], axis=0)[np.newaxis,:]
        corpus = corpus_raw - corpus_mean
        corpus = np.array([matutils.unitvec(row) for row in corpus])

        corpus_dict = {}
        corpus_dict['raw_corpus'] = corpus_raw
        corpus_dict['corpus'] = corpus
        corpus_dict['mean'] = corpus_mean
        return corpus_dict
        
        
