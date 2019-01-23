import sys
import sklearn.gaussian_process as gausspproc
import subprocess
import pickle
import numpy as np
import os
from utils import get_best_params



SAVEDIR='.../results/training/experiment01/t_{}/{}/{}'
MODEL=''  # 'kmeans' or 'stochastickmeans' or 'random_emb'
CORPUS='' # '20NewsGroupCachopo' or 'RCV1'
BATCH=''  
NRUNS=1
MINITER=200
MAXITER=1000
LRATE=0.001


for run in range(NRUNS):
    for NTOPICS in [5, 10, 20, 50, 100]:
        savedir = os.path.join(SAVEDIR.format(str(NTOPICS),MODEL, CORPUS), str(run))
        cmd = [ 'python', '.../topics/run_model.py','--model', MODEL, '--corpus',CORPUS,  '--savedir', savedir, '--t', str(NTOPICS), '--lr', str(LRATE), '--gamma', str(.99),'--maxiter',str(MAXITER), '--miniter',str(MINITER), '--bs', str(BATCH), '--case', ""]

        p = subprocess.Popen(cmd)
        p.wait()
        




        






