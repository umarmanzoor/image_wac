
from __future__ import division
import pandas as pd
import numpy as np
import gzip
import cPickle as pickle
import json

import scipy.stats
import nltk
from nltk.tag.perceptron import PerceptronTagger
from time import strftime
import re
import sys
sys.path.append('../TrainModels')
sys.path.append('../Utils')
from utils import filter_by_filelist, icorpus_code
from train_model import STOPWORDS, is_relational, create_word2den, make_train
from apply_model import *

import os

def preproc(utterance):
    utterance = re.sub('[\.\,\?;]+', '', utterance)
    return utterance.lower()
preproc_vec = np.vectorize(preproc)

tagger = PerceptronTagger()
def postag(refexp):
    return nltk.tag._pos_tag(nltk.word_tokenize(refexp), None, tagger)

outfilename = 'EvalOut/results-clef2.pklz'

# if os.path.isfile(outfilename):
#     print 'Outfile (%s) exists. Better check before I overwrite anything!' % (outfilename)
#     exit()

results = []

print strftime("%Y-%m-%d %H:%M:%S")
print 'Loading up data. This may take some time.'

### Load up basic common data

## Corpora, Features

X = np.load('/home/umarmanzoor/image_wac/ExtractFeats/ExtrFeatsOut/saiapr.npz')
X = X['arr_0']

#*************************************************
## Reading Referit generated using our Annotation
#*************************************************
referitpath = '../Data/RefExps/SAIAPR/ReferIt/clef_annotated/AnnotatedClef.txt'

srefdf = pd.read_csv(referitpath, sep='~', names=['ID', 'refexp', 'regionA', 'regionB'])
srefdf['file'] = srefdf['ID'].apply(lambda x: int(x.split('.')[0].split('_')[0]))
srefdf['region'] = srefdf['ID'].apply(lambda x: int(x.split('.')[0].split('_')[1]))
srefdf['refexp'] = preproc_vec(srefdf['refexp'])

srefdf['i_corpus'] = icorpus_code['saiapr']
srefdf['r_corpus'] = 'referit'
srefdf['image_id'] = srefdf['file']
srefdf['region_id'] = srefdf['region']
srefdf['rex_id'] = srefdf.index.tolist()

srefdf['tagged'] = srefdf['refexp'].apply(postag)

srefdf = srefdf[['i_corpus', 'image_id', 'region_id', 'r_corpus',
               'rex_id', 'refexp', 'tagged']]

##*************************************************
## Annotated CLEF TestData
##*************************************************
test_dataset_p = 'clef_test.txt'
saiapr_clef_files = np.loadtxt(test_dataset_p, dtype=int)

ssplit90 = {
    'test': list(saiapr_clef_files)
}

## Bounding box definitions

with gzip.open('../Preproc/PreProcOut/saiapr_bbdf.pklz', 'r') as f:
    s_bbdf = pickle.load(f)

print strftime("%Y-%m-%d %H:%M:%S")
print 'Off we go.'


## Run the Clef Evaluation:

execfile('EvalDefs/clefdata.py')

with gzip.open(outfilename, 'w') as f:
    pickle.dump(results, f)
