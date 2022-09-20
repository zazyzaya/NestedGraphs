import glob
import os 
import socket 

# Because sklearn can't be trusted with this for some reason
# this has to be done before they're imported
os.environ['MKL_NUM_THREADS'] = '1'

import torch
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor as LOF 
from sklearn.svm import OneClassSVM 
from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score, \
    precision_score, recall_score, f1_score, accuracy_score 
import pandas as pd 

# Depending on which machine we're running on 
if socket.gethostname() == 'colonial0':
    HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/'

# Note, this is running over sshfs so it may be slower to load
# may be worth it to make a local copy? 
elif socket.gethostname() == 'orion.ece.seas.gwu.edu':
    HOME = '/home/isaiah/code/NestedGraphs/inputs/'

WORKERS = 16
torch.set_num_threads(8)

def get_tr(day=23):
    embs = glob.glob(HOME+'Sept%d/benign/tgat_emb_clms*' % day)
    zs = []
    for e in embs:
        data = torch.load(e)
        zs.append(data['zs'][data['proc_mask']])

    return torch.cat(zs,dim=0).cpu()

def get_te(day=23):
    embs = glob.glob(HOME+'Sept%d/mal/tgat_emb_clms*' % day)
    zs,y = [],[]

    for e in embs: 
        data = torch.load(e)
        zs.append(data['zs'][data['proc_mask']])
        y.append(data['y'])

    return torch.cat(zs,dim=0).cpu(), torch.cat(y,dim=0).cpu().clamp(0,1)

def get_stats():
    return {
        'Pr': [], 'Re': [],
        'F1': [], 'Ac': [],
        'AUC': [], 'AP': []
    }

def add_stats(stats, preds, y_hat, y, verbose=True):
    stats['Pr'].append(precision_score(y, y_hat))
    stats['Re'].append(recall_score(y, y_hat))
    stats['F1'].append(f1_score(y, y_hat))
    stats['Ac'].append(accuracy_score(y, y_hat))
    stats['AUC'].append(auc_score(y, preds))
    stats['AP'].append(ap_score(y, preds))

    if not verbose:
        return 

    for k,v in stats.items():
        print(k,v[-1])


def test_lof(params, day=23):
    lof = LOF(**params, novelty=True, n_jobs=WORKERS)
    tr = get_tr(day)

    stats = get_stats() 

    print("Fitting",tr.size(0),'samples')
    lof.fit(tr)

    # Need to flip and convert to (0,1)
    # by default -1 means outlier, 1 is inlier
    te,y = get_te(day)
    y_hat = -lof.predict(te)
    y_hat[y_hat==-1] = 0

    # Same here, smaller (negative) numbers are outliers, 
    # so negating means inliers are close to 1 outliers tend to inf
    preds = -lof.score_samples(te)
    add_stats(stats, preds, y_hat, y)
    
    return pd.DataFrame(stats)

def test_lof_all(params, day=23):
    lof = LOF(**params, n_jobs=WORKERS)
    tr = get_tr(day)
    te,y = get_te(day)

    stats = get_stats() 

    # LOF computes on all available data
    x = torch.cat([tr,te])
    y = torch.cat([torch.zeros(tr.size(0)),y])

    print("Fitting",x.size(0),'samples')
    
    # Need to flip and convert to (0,1)
    # by default -1 means outlier, 1 is inlier
    y_hat = -lof.fit_predict(x)
    y_hat[y_hat==-1] = 0

    # Same here, smaller (negative) numbers are outliers, 
    # so negating means inliers are close to 1 outliers tend to inf
    preds = -lof.negative_outlier_factor_

    add_stats(stats, preds[tr.size(0):], y_hat[tr.size(0):], y[tr.size(0):])
    
    return pd.DataFrame(stats)


def test_ocsvm(params, day=23):
    svm = OneClassSVM(**params)
    tr = get_tr(day)

    stats = get_stats() 
    print("Fitting",tr.size(0),'samples')
    svm.fit(tr)

    # Need to flip and convert to (0,1)
    # by default -1 means outlier, 1 is inlier
    te,y = get_te(day)
    y_hat = -svm.predict(te)
    y_hat[y_hat==-1] = 0

    # Smaller numbers correlate to greater chance of outlier
    # for now, just using inverse
    preds = 1/(svm.score_samples(te)+1e-9)
    add_stats(stats, preds, y_hat, y)
    return pd.DataFrame(stats)


if __name__ == '__main__':
    params = dict()

    contaminate = [1e-7,1e-5,1e-4,1e-3]
    for c in contaminate:
        params['n_neighbors'] = 50
        params['contamination'] = c
        
        stats = test_lof(params)
        with open(HOME+'../results/lof.txt', 'a') as out_f:
            out_f.write('Neighbors: 50\tContamination: %d\n' % c)
            stats.to_csv(out_f, sep='\t')
            stats.mean().to_csv(out_f, sep='\t')
            stats.sem().to_csv(out_f, sep='\t')

    '''
    # Takes forever; run overnight.
    out_f  = open(HOME+'../results/oc-svm.txt', 'w+')
    out_f.close() 

    params['kernel'] = 'poly'
    degree = [1,2,3,4]
    coef = [0., 0.01, 0.1, 1]
    
    for d in degree:
        for c in coef:
            params['coef0'] = c
            params['degree'] = d

            stats = test_ocsvm(params)
            print(stats)
            with open(HOME+'../results/lof.txt', 'a') as out_f:
                out_f.write('\n\ncoef0:%f, Degree:%d\n' % (c,d))
                stats.to_csv(out_f, sep='\t')
                stats.mean().to_csv(out_f, sep='\t')
                stats.sem().to_csv(out_f, sep='\t')
    '''