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
import collections
from gensim.parsing.preprocessing import preprocess_string, \
strip_multiple_whitespaces,strip_non_alphanum, strip_numeric, \
strip_punctuation, strip_tags, strip_short
from utils import set_up_dir
import subprocess
from nltk.corpus import stopwords
import pandas
import re

stop_words = stopwords.words('english')
stop_words = stop_words 


        
class NYTCorpus(object):
    def __init__(self, root_dir):
        self.name = 'NYT'
        self.json_articles = os.path.join(root_dir, self.name,  'articles-search-1990-2016.json')
        self.json_paragraphs = os.path.join(root_dir, self.name,  'paragraphs-1990-2016.json')
        self.json_proc = os.path.join(root_dir, self.name,  'data-1990-2016.json')


        self.txt_source_train = None
        self.txt_source_test = None
        self.root_dir = root_dir
        
        
    def create_clean_data(self):
        df_articles = pandas.read_json(self.json_articles)
        df_paragraphs = pandas.read_json(self.json_paragraphs)

        
        df_articles_clean = df_articles.drop(['snippet', 'word_count',  'desk', 'lead', 'author', 'abstract'], axis=1)
        df_articles_clean = df_articles_clean.dropna(subset=['section'])


        # Adding paragraph to dataframe
        c = []
        df_articles_clean['paragraphs'] = None

        for k in df_articles_clean['id']:
            text = list(df_paragraphs[df_paragraphs['id']==k]['paragraphs'])
            c.append(text)

        df_articles_clean['paragraphs'] = c
        df_articles_clean['year'] = df_articles_clean['date'].dt.year
        df_articles_clean['month'] = df_articles_clean['date'].dt.month
        df_articles_clean['day'] = df_articles_clean['date'].dt.day

        # Remove empty paragraph items
        df_articles_filtered = df_articles_clean[[False if len(np.squeeze(np.array(x)).tolist())==0 else True for x in df_articles_clean['paragraphs']]]
        
        df_articles_filtered.to_json(self.json_proc)

        
    def get_timewindow(self, time_dict):
        years = list(time_dict.keys())
        y_min, y_max = np.min(years), np.max(years)
        m_min = np.min(list(time_dict[y_min]))
        m_max = np.max(list(time_dict[y_max]))
        
        return (m_min, y_min, m_max, y_max)
    
    def create_static(self):
        subdir = 'static'
        target_dir = os.path.join(self.root_dir, self.name, subdir)                     
        
        df = pandas.read_json(self.json_proc) # load
        
        window_dirs = []
        total = 0.
        
        save_dir = target_dir
        set_up_dir(save_dir)
        
        df_train=df.sample(frac=0.8,random_state=42)
        df_test=df.drop(df_train.index)
        
        assert len(df_train) + len(df_test) == len(df)

        df_train.to_json(os.path.join(save_dir, 'train.json'))
        df_test.to_json(os.path.join(save_dir, 'test.json'))
        
        return save_dir    
        
    def create_timewindows(self, time_dict, resolution = 'monthly'):
        
        subdir = '{:02}{:.0f}_{:02}{:.0f}'.format(*self.get_timewindow(time_dict))
        target_dir = os.path.join(self.root_dir, self.name, subdir, resolution)                     
        
        df = pandas.read_json(self.json_proc) # load
        
        window_dirs = []
        total = 0.
        for year, months in time_dict.items():
            
            df_year = df[df['year'] == year]

            if resolution == 'monthly':
                for month in months:
                    save_dir = os.path.join(target_dir, '{:02}{:.0f}'.format(month, year))
                    set_up_dir(save_dir)

                    df_tmp = df_year[df_year['month'] == month]
                    print('{}\t {} \t {}'.format(year, month, len(df_tmp)))
                    total += len(df_tmp)

                    df_train=df_tmp.sample(frac=0.8,random_state=42)
                    df_test=df_tmp.drop(df_train.index)

                    assert len(df_train)+len(df_test) == len(df_tmp)     

                    window_dirs.append(save_dir)
                    
            elif resolution == 'yearly':
                save_dir = os.path.join(target_dir, '{:.0f}'.format(year))
                set_up_dir(save_dir)
                
                print('{}\t {}'.format(year, len(df_year)))
                total += len(df_year)

                df_train=df_year.sample(frac=0.8,random_state=42)
                df_test=df_year.drop(df_train.index)
                
                window_dirs.append(save_dir)
                
            else:
                raise
                
            df_train.to_json(os.path.join(save_dir, 'train.json'))
            df_test.to_json(os.path.join(save_dir, 'test.json'))

                        
        return window_dirs
                                      
 
        
    def get_train(self):
        df_train = pandas.read_json(self.txt_source_train)
        
        for _id, row in df_train.iterrows():
    
            id_hash = row['id']
            paragraph = row['paragraphs']      
            text = ' '.join(list(itertools.chain(*paragraph)))
            date = re.sub('[^0-9]', '', str(row['date']).split()[0])
            
            words = [w.lower() for w in text.strip().split()]    
            words = [re.sub('[^a-zA-Z0-9]', '', w) for w in words]   
            words = [w for w in words if w not in stop_words]
            
            if len(words) > 0:            
                yield gensim.models.doc2vec.TaggedDocument(words = words, tags = ['_'.join(['train', date, str(_id), str(id_hash)])])
                
                
 
    def get_test(self):
        df_test = pandas.read_json(self.txt_source_test)
        
        for _id, row in df_test.iterrows():
    
            id_hash = row['id']
            paragraph = row['paragraphs']      
            text = ' '.join(list(itertools.chain(*paragraph)))

            date = re.sub('[^0-9]', '', str(row['date']).split()[0])
            
            words = [w.lower() for w in text.strip().split()]    
            words = [re.sub('[^a-zA-Z0-9]', '', w) for w in words]   
            words = [w for w in words if w not in stop_words]
            
            if len(words) > 0:            
                yield gensim.models.doc2vec.TaggedDocument(words = words, tags = ['_'.join(['test', date, str(_id), str(id_hash)])])
         
        
        
        
        
