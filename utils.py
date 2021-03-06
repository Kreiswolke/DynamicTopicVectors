import requests
import subprocess
import os
import gensim
import pickle
import numpy as np
import re
import codecs


def write_topic_files(res, dst, vocabulary, maxtops=10,remove_mean = False, word_model_file = None):
    if word_model_file and isinstance(word_model_file, str):
        word_model = get_wordmodel(word_model_file)
    elif word_model_file:
        word_model = word_model_file
    else:
        raise

    T = res['topics']
    n_topics = res['n_topics']
    assert np.isclose(np.mean(np.linalg.norm(T,axis=1)), 1.)
    
    if remove_mean == True:
    	T,T_mean = remove_mean(T)
    
    index = gensim.similarities.SparseMatrixSimilarity(T, num_features = 300) # N_topics x N_docs/num_best x (topic_id,weight)
    sims = index[T]

    with open(dst, 'wb') as f:
        res_topics = {}
        for i,t in enumerate(T):
            print(i)
            W = []
            E = []
            out = word_model.similar_by_vector(t, topn=5)
                
            for word in word_model.similar_by_vector(t, topn=100000):
                if word[0] in vocabulary and len(W) < maxtops:
                    W.append(word[0])
                    E.append(word[1])
                else:
                    pass
                
            res_topics[i] = {'words' : W, 'weights' : E}

            line = ' '.join(W) + '\n'
            print(line)
            f.write(line.encode('utf-8'))
            
    return res_topics


def read_vocab_dict(vocab_url):
    fin = open(vocab_url, 'rb')
    vocab = {}
    i = 0
    while True:
        line = fin.readline()
        if not line:
            break
        out = line.split()
        word, count = out[0].strip(), int(out[1].strip())
        vocab[i] = word.decode('utf8')
        i+=1
    print(i)
    fin.close()
    return vocab

def set_up_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def download_file(url, target):
    from tqdm import tqdm
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    with open(target, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading file') as progress_bar:
            for data in r.iter_content(32*1024):
                f.write(data)
                progress_bar.update(32*1024)


def get_wordmodel(wordmodel_file):
    if wordmodel_file.endswith('txt'):
        model = gensim.models.KeyedVectors.load_word2vec_format(wordmodel_file)
    elif wordmodel_file.endswith('bin'): 
        model = gensim.models.KeyedVectors.load_word2vec_format(wordmodel_file, binary=True)
    else:
        print('Unknown word model type', wordmodel_file)
        raise 

    return model


def get_data(case, root_dir, valid_dict=True):
 
    doc_dict_train = pickle.load(open(os.path.join(root_dir, case, 'proc', 'train_doc_dict.p'),'rb'))

    doc_dict_test = pickle.load(open(os.path.join(root_dir, case, 'proc','testsplit_doc_dict.p'),'rb'))
   
    try:
        vocab =  pickle.load(open(os.path.join(root_dir, case, 'proc_onehot','vocab.pkl'),'rb'))
    except: 
        vocab = None

    if valid_dict==True:    

        doc_dict_valid = pickle.load(open(os.path.join(root_dir, case, 'proc','valid_doc_dict.p'),'rb'))

        return doc_dict_train, doc_dict_test, doc_dict_valid, vocab
    else:
        return doc_dict_train, doc_dict_test, vocab
    

class InitMyCorpus(object):
    def __init__(self, case, corpus_dir):
        self.case = case
        self.data_dir = corpus_dir

class MyCorpus(gensim.corpora.TextCorpus):
 
    def get_texts(self): 
        for k,v in self.input.items():     
            yield v['words']
            
def get_weights(dir_):
    W0 = pickle.load(open(dir_,'rb'))['topics']
    return W0




