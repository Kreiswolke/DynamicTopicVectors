import sys
import sklearn.gaussian_process as gausspproc
import subprocess
import pickle
import numpy as np
import os


def run_cmd(cmd):
    print(' '.join(cmd))
    p = subprocess.Popen(cmd)
    p.wait()

# Specify 
SAVEDIR='results/experiment01/t_{}/{}/{}'
CORPUS='NYT' 
BATCH=200
MINITER=20
MAXITER=1000
NRUNS=1
NTOPICS = [50]
LRATE=0.005
NRUNS=1
 

stamps = ['1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']
interval = '01{}_06{}'.format(stamps[0], stamps[-1])
MODEL='DTV' 



for n_topic in NTOPICS:
    for run in range(NRUNS):
        for stamp in stamps:
            
            savedir = os.path.join(SAVEDIR.format(str(n_topic),MODEL, CORPUS), str(run), stamp)
            CORPUS_CASE = os.path.join(CORPUS, interval, 'yearly', stamp)

            if stamp == stamps[0]:
             # Random topic intiialization
                cmd = [ 'python', 'run_model.py','--model', MODEL, '--corpus',CORPUS_CASE,  '--savedir', savedir, '--t', str(n_topic), '--lr', str(LRATE), '--gamma', str(.99),'--maxiter',str(MAXITER), '--miniter',str(MINITER), '--bs', str(BATCH), '--case', "", '--eta', str(0.001)]
                run_cmd(cmd)
             # Initialization from previous run 
            else:
                weightdir = os.path.join(SAVEDIR.format(str(n_topic),MODEL, CORPUS), str(run), last_stamp, 'res_n_{}.p'.format(n_topic))
                cmd = [ 'python', 'run_model.py','--model', MODEL, '--corpus',CORPUS_CASE,  '--savedir', savedir, '--t', str(n_topic), '--lr', str(LRATE), '--gamma', str(.99),'--maxiter',str(MAXITER), '--miniter',str(MINITER), '--bs', str(BATCH), '--case', "", '--weight', weightdir , '--eta', str(0.001)]
                run_cmd(cmd)

            last_stamp=stamp






