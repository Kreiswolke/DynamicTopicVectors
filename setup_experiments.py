from proc_datasets import  CreateItem2Vector, CorpusProcessor, CorpusOneHotProcessor
import subprocess
from utils import set_up_dir
import os
from datasets import RCV1, TwentyNewsGroup
from utils import get_data

WORDMODEL_DIR = '/topics/wordmodel/GloVe/'
ROOT_DATA_DIR= '/topics/data/'
set_up_dir(WORDMODEL_DIR)


#Download wordmodel
p=subprocess.Popen(['wget', '-c', '--no-check-certificate', 'nlp.stanford.edu/data/glove.6B.zip'], cwd=WORDMODEL_DIR)
p.wait()
p=subprocess.Popen(['unzip', 'glove.6B.zip'], cwd=WORDMODEL_DIR)
p.wait()
p = subprocess.Popen(['python', '-m', 'gensim.scripts.glove2word2vec', '--input', 'glove.6B.300d.txt', '--output', 'glove2word2vec300d.txt'], cwd=WORDMODEL_DIR)
p.wait()

# Download 20NewsGroup cachopo and RCV1 dataset

corpus = TwentyNewsGroup(ROOT_DATA_DIR)
corpus.download()

corpus = RCV1(ROOT_DATA_DIR)
corpus.download()



for corpus in [RCV1(ROOT_DATA_DIR)]:

	corpus_name = corpus.name
	# Process datasetet
	WORDMODEL_FILE = os.path.join(WORDMODEL_DIR, 'glove2word2vec300d.txt')
	PROC_DATA_DIR = os.path.join(ROOT_DATA_DIR, corpus_name, 'proc')
	set_up_dir(PROC_DATA_DIR)

	processor = CorpusProcessor(corpus, WORDMODEL_FILE, PROC_DATA_DIR)
	processor.__call__()
	processor.split()


        





                

