import argparse
import sys

import pickle 
import torch 
import datetime as dt
from zoneinfo import ZoneInfo
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from torch.distributions import Normal

from graph_utils import propagate_labels

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'
torch.set_num_threads(8)

fmt_ts = lambda x : dt.datetime.fromtimestamp(x).astimezone(
    ZoneInfo('Etc/GMT+4')
).isoformat()[11:-6]

def test_emb_input(zs, graph, disc, day):
    labels = propagate_labels(graph,day).clamp(0,1)
    
    with torch.no_grad():
        disc.eval()
        preds = torch.sigmoid(disc(zs, graph).nan_to_num())

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)
    labels = labels[idx].clamp(0,1)

    auc_score = auc(labels, vals)
    ap_score = ap(labels, vals)

    top_k = [50, 100, 150, 200]
    pr = {}
    for k in top_k: 
        preds = torch.zeros(vals.size())
        preds[:k] =1.
        
        p = precision_score(labels, preds)
        r = recall_score(labels, preds)

        pr[k] = (p,r)

    return auc_score, ap_score, pr


def get_metrics(preds, labels, cutoff=None):
    auc_score = auc(labels, preds)
    ap_score = ap(labels, preds)

    top_k = [50, 100, 150, 200]
    pr = {}
    for k in top_k: 
        preds = torch.zeros(preds.size())
        preds[:k] =1.
        
        p = precision_score(labels, preds)
        r = recall_score(labels, preds)

        pr[k] = (p,r)

    if cutoff is None:
        return auc_score, ap_score, pr

    y_hat = torch.zeros(preds.size())
    y_hat[preds > cutoff] = 1. 

    p = precision_score(labels, y_hat)
    r = recall_score(labels, y_hat)
    a = accuracy_score(labels, y_hat)
    f1 = f1_score(labels, y_hat)

    return auc_score, ap_score, p, r, a, f1, pr

@torch.no_grad()
def score_many(gids, disc, day, cutoff=None):
    disc.eval() 

    preds = []
    labels = []

    for gid in gids: 
        z = torch.load(HOME+'inputs/Sept%d/mal/emb%d.pkl' % (day, gid))
        with open(HOME+'inputs/Sept%d/mal/graph%d.pkl' % (day,gid), 'rb') as f:
            graph = pickle.load(f)
        
        labels.append(propagate_labels(graph, day).clamp(0,1))
        preds.append(disc(z, graph))

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    return get_metrics(preds, labels, cutoff)
