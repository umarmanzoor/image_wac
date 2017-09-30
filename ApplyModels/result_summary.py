from __future__ import division
import pandas as pd
import numpy as np
import gzip
import cPickle as pickle

import matplotlib
import matplotlib.pyplot as plt

pd.set_option('display.precision', 2)

def is_iou_over_threshold_top_n(row, threshold=0.5, n=1, random=False):
    if np.isnan(row['nob']):
        return np.nan
    if random:
        return np.any(np.array(row['ious'])[np.random.choice(range(len(row['ious'])), n)] > threshold)
    return np.any(np.array(row['ious'])[row['rnk']][:n] > threshold)

with gzip.open('EvalOut/results-clef2.pklz', 'r') as f:
    results = pickle.load(f)

def mrr_f(series):
    return np.mean(series.apply(lambda x:(1/x)))

def acc_f(series):
    return np.count_nonzero(np.nan_to_num(series.tolist())) / len(series)

def summarise_rdf(rdf):
    this_row = {}
    # for the full data frame
    this_row['acc-full'] = acc_f(rdf['suc'])
    this_row['mrr-full'] = mrr_f(rdf['rnk'])
    this_row['arc-full'] = rdf['cov'].mean()
    this_row['rnd-full'] = mrr_f(rdf['nob'])
    # for the ones where at least one word was known
    nz_rdf = rdf.query('cov > 0')
    this_row['>0 wrcov'] = len(nz_rdf) / len(rdf)
    this_row['acc->0wc'] = acc_f(nz_rdf['suc'])
    this_row['mrr->0wc'] = mrr_f(nz_rdf['rnk'])
    this_row['arc->0wc'] = nz_rdf['cov'].mean()
    this_row['rnd->0wc'] = mrr_f(nz_rdf['nob'])
    # binned by refexp length
    lens = nz_rdf['refexp'].apply(lambda x: len(x.split()))
    this_bin = nz_rdf[(lens > 0) & (lens <= 2)]
    this_row['acc-b1-2'] = acc_f(this_bin['suc'])
    this_row['12%'] = len(this_bin) / len(nz_rdf)
    this_bin = nz_rdf[(lens > 2) & (lens <= 4)]
    this_row['acc-b3-4'] = acc_f(this_bin['suc'])
    this_row['34%'] = len(this_bin) / len(nz_rdf)
    this_bin = nz_rdf[(lens > 4) & (lens <= 6)]
    this_row['acc-b5-6'] = acc_f(this_bin['suc'])
    this_row['56%'] = len(this_bin) / len(nz_rdf)
    return this_row

def summarise_rdf_rprop(rdf):
    this_row = {}
    # for the full data frame
    this_row['RP@1-full'] = acc_f(rdf.apply(is_iou_over_threshold_top_n, axis=1))
    this_row['RP@10-full'] = acc_f(rdf.apply(lambda x:is_iou_over_threshold_top_n(x, n=10),
                                           axis=1))
    this_row['arc-full'] = rdf['cov'].mean()
    this_row['rnd-full'] = acc_f(rdf.apply(lambda x:is_iou_over_threshold_top_n(x, random=True),
                                           axis=1))
    # for the ones where at least one word was known
    nz_rdf = rdf.query('cov > 0')
    this_row['>0 wrcov'] = len(nz_rdf) / len(rdf)
    this_row['RP@1->0wc'] = acc_f(nz_rdf.apply(is_iou_over_threshold_top_n, axis=1))
    this_row['RP@10->0wc'] = acc_f(nz_rdf.apply(lambda x:is_iou_over_threshold_top_n(x, n=10),
                                               axis=1))
    this_row['arc->0wc'] = nz_rdf['cov'].mean()
    this_row['rnd->0wc'] = acc_f(nz_rdf.apply(lambda x:is_iou_over_threshold_top_n(x, random=True),
                                           axis=1))
    return this_row


index = []
rows = []
for model, rdf in results:
    if 'rprop' in model:
        continue
    else:
        index.append(model)
        this_resdict = summarise_rdf(rdf)
        this_resdict['%tst'] = 1.0
        rows.append(this_resdict)
        index.append(model + '; NR')
        rdf_norel = rdf.query('is_rel == False')
        this_resdict = summarise_rdf(rdf_norel)
        this_resdict['%tst'] = len(rdf_norel) / len(rdf)
        rows.append(this_resdict)

collected_columns = {}
for this_row in rows:
    for this_key, this_val in this_row.items():
        this_list = collected_columns.get(this_key, list())
        this_list.append(this_val)
        collected_columns[this_key] = this_list

full_df = pd.DataFrame(collected_columns, index=index)
full_df = full_df[['%tst', 'acc-full', 'mrr-full', 'arc-full', 'rnd-full', '>0 wrcov',
                   'acc->0wc', 'mrr->0wc', 'arc->0wc', 'rnd->0wc',
                   'acc-b1-2', '12%', 'acc-b3-4', '34%', 'acc-b5-6', '56%']]

pd.set_option('display.float_format', '{:.2f}'.format)
print(full_df)