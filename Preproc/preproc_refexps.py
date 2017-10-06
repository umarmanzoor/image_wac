# coding: utf-8
from __future__ import division

import json
import numpy as np
import pandas as pd
import cPickle as pickle
import gzip
import re
import datetime

import nltk
from nltk.tag.perceptron import PerceptronTagger

import sys
sys.path.append('/home/umarmanzoor/image_wac/Utils')
from Utils.utils import icorpus_code


def preproc(utterance):
    utterance = re.sub('[\.\,\?;]+', '', utterance)
    return utterance.lower()
preproc_vec = np.vectorize(preproc)

tagger = PerceptronTagger()
def postag(refexp):
    return nltk.tag._pos_tag(nltk.word_tokenize(refexp), None, tagger)


now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print 'starting to preprocess...'
print now

## ReferIt SAIAPR
referitpath = '../Data/RefExps/SAIAPR/ReferIt/RealGames.txt'

refdf = pd.read_csv(referitpath, sep='~', names=['ID', 'refexp', 'regionA', 'regionB'])
refdf['file'] = refdf['ID'].apply(lambda x: int(x.split('.')[0].split('_')[0]))
refdf['region'] = refdf['ID'].apply(lambda x: int(x.split('.')[0].split('_')[1]))
refdf['refexp'] = preproc_vec(refdf['refexp'])

refdf['i_corpus'] = icorpus_code['saiapr']
refdf['r_corpus'] = 'referit'
refdf['image_id'] = refdf['file']
refdf['region_id'] = refdf['region']
refdf['rex_id'] = refdf.index.tolist()

refdf['tagged'] = refdf['refexp'].apply(postag)

refdf = refdf[['i_corpus', 'image_id', 'region_id', 'r_corpus', 
               'rex_id', 'refexp', 'tagged']]


# load and write out the splits on SAIAPR as used by Berkeley group (50/50)
b_splits_train_p = '../Data/Images/SAIAPR/Berkeley_rprops/referit_trainval_imlist.txt'
b_splits_test_p = '../Data/Images/SAIAPR/Berkeley_rprops/referit_test_imlist.txt'

saiapr_train_files = np.loadtxt(b_splits_train_p, dtype=int)
saiapr_test_files = np.loadtxt(b_splits_test_p, dtype=int)

saiapr_berkeley_splits = {
    'test': list(saiapr_test_files),
    'train': list(saiapr_train_files)
}


with open('PreProcOut/saiapr_berkeley_10-10_splits.json', 'w') as f:
    json.dump(saiapr_berkeley_splits, f)


# create a 90/10 split as well, to have more training data
saiapr_full = list(saiapr_train_files) + list(saiapr_test_files)
#saiapr_train_90 = list(saiapr_train_files) + list(saiapr_test_files)[:8000]
#saiapr_test_90 = list(saiapr_test_files)[8000:]

saiapr_no_splits = {
    'train': saiapr_full
}
with open('PreProcOut/saiapr__splits.json', 'w') as f:
    json.dump(saiapr_no_splits, f)

### Done!

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print 'done with the assembling...'
print now

print 'example rows:'
print refdf.head(1)
print 'refdf: %d' % (len(refdf))


## Well ok, we should probably write out to disk as well:

with gzip.open('PreProcOut/saiapr_refdf.pklz', 'w') as f:
    pickle.dump(refdf, f)

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
print '.. and done!'
print now