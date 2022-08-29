import torch
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor as LOF 
from sklearn.svm import OneClassSVM 
from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score, \
    precision_score, recall_score, f1_score, accuracy_score 
import pandas as pd 

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/'
WORKERS = 8

def get_tr(day=23):
    return torch.load(HOME+'Sept%d/benign/all_proc_embs.pkl' % day)

def get_te(day=23):
    return  torch.load(HOME+'Sept%d/mal/all_proc_embs.pkl' % day), \
            torch.load(HOME+'Sept%d/mal/ys.pkl' % day)

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
    lof = LOF(**params, n_jobs=WORKERS)
    #tr = get_tr(day)
    te,y = get_te(day)

    stats = get_stats() 

    # LOF computes on all available data
    #x = torch.cat([tr,te])
    #y = torch.cat([torch.zeros(tr.size(0)),y])
    x = te 

    print("Fitting",x.size(0),'samples')
    
    # Need to flip and convert to (0,1)
    # by default -1 means outlier, 1 is inlier
    y_hat = -lof.fit_predict(x)
    y_hat[y_hat==-1] = 0

    # Same here, smaller (negative) numbers are outliers, 
    # so negating means inliers are close to 1 outliers tend to inf
    preds = -lof.negative_outlier_factor_

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

    add_stats(stats, preds, y_hat, y)
    
    return pd.DataFrame(stats)


if __name__ == '__main__':
    params = dict(
        n_neighbors=20,
        leaf_size=1000
    )

    neighbors = [20,100,250,500,1000,2500,5000]

    out_f  = open(HOME+'../results/lof.txt', 'w+')
    out_f.close() 
    for n in neighbors:
        params['n_neighbors'] = n
        
        stats = test_lof(params)
        with open(HOME+'../results/lof.txt', 'a') as out_f:
            out_f.write('Neighbors: %d\n' % n)
            stats.to_csv(out_f, sep='\t')
            stats.mean().to_csv(out_f, sep='\t')
            stats.sem().to_csv(out_f, sep='\t')


    for n in neighbors:
        params['n_neighbors'] = n

        stats = test_lof_all(params)
        with open(HOME+'../results/lof.txt', 'a') as out_f:
            out_f.write('\n\n(Full) Neighbors: %d\n' % n)
            stats.to_csv(out_f, sep='\t')
            stats.mean().to_csv(out_f, sep='\t')
            stats.sem().to_csv(out_f, sep='\t')