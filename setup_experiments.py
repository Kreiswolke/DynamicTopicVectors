from proc_datasets import CorpusProcessor
import subprocess
from utils import set_up_dir, get_data
import os
from datasets import NYTCorpus
import os 

dtv_path = os.path.dirname(os.path.realpath(__file__))

WORDMODEL_DIR =  os.path.join(dtv_path, 'wordmodel', 'GloVe')
ROOT_DATA_DIR= os.path.join(dtv_path, 'data')
set_up_dir(WORDMODEL_DIR)



#Download wordmodel
if False:
	p=subprocess.Popen(['wget', '-c', '--no-check-certificate', 'nlp.stanford.edu/data/glove.6B.zip'], cwd=WORDMODEL_DIR)
	p.wait()
	p=subprocess.Popen(['unzip', 'glove.6B.zip'], cwd=WORDMODEL_DIR)
	p.wait()
	p = subprocess.Popen(['python', '-m', 'gensim.scripts.glove2word2vec', '--input', 'glove.6B.300d.txt', '--output', 'glove2word2vec300d.txt'], cwd=WORDMODEL_DIR)
	p.wait()

# Process NYT dataset

time_dict = {i: range(1,13) for i in range(1991, 2016)}
time_dict.update({2016: range(1,7)})

corpus = NYTCorpus(ROOT_DATA_DIR)
corpus_name = corpus.name
#corpus.create_clean_data()
window_dirs = corpus.create_timewindows(time_dict, resolution = 'yearly')

WORDMODEL_FILE = os.path.join(WORDMODEL_DIR, 'glove2word2vec300d.txt')

for ROOT_DATA_DIR in window_dirs:
	corpus.txt_source_train = os.path.join(ROOT_DATA_DIR, 'train.json')
	corpus.txt_source_test = os.path.join(ROOT_DATA_DIR, 'test.json')

	PROC_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'proc')
	set_up_dir(PROC_DATA_DIR)

	processor = CorpusProcessor(corpus, WORDMODEL_FILE, PROC_DATA_DIR)
	processor.__call__()
	processor.split()



        





                

