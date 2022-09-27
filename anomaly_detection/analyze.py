import glob
import pickle
import sys 

from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score, \
    precision_score, recall_score, f1_score, accuracy_score 
import torch 

from conventional import get_te, get_stats, add_stats, HOME

def test_one_pure_nodes(fname, y):
    with open(fname, 'rb') as f:
        record = pickle.load(f)

    # This is just the format I saved them in
    preds, thresh = record[:-1], record[-1]
    y_hat = torch.zeros(preds.shape)
    y_hat[preds > thresh] = 1.

    stats = get_stats()
    return add_stats(stats, preds, y_hat, y)

def test_all_pure_nodes():
    _,y = get_te()
    for fname in glob.glob('ocsvm_out/*'):
        params = fname.split('/')[-1].split('_')
        print(*params)
        test_one_pure_nodes(fname, y)
        print()

def get_cc_data(day=23):
    embs = glob.glob(HOME+'Sept%d/mal/tgat_emb_clms*' % day)
    zs,y,cc = [],[],[]

    for e in embs: 
        data = torch.load(e, map_location=torch.device('cpu'))
        y.append(data['y'])
        cc.append(data['ccs'])

    return y,cc

def test_one_cc(fname, y,cc):
    with open(fname, 'rb') as f:
        record = pickle.load(f)

    # This is just the format I saved them in
    preds, thresh = record[:-1], record[-1]
    cc_preds, cc_y = [],[]

    last_cnt = 0
    for i in range(len(y)):
        y_i = y[i]
        cc_i = cc[i]
        
        my_preds = preds[last_cnt : last_cnt+y_i.size(0)]
        last_cnt += y_i.size(0)

        # Each cc is scored as the maximum of its nodes
        for component in cc_i:
            cc_preds.append(my_preds[component].max())
            cc_y.append(y_i[component].sum().clamp(0,1))

    preds = torch.tensor(cc_preds)
    y_hat = torch.zeros(preds.size())
    y = torch.stack(cc_y)
    y_hat[preds > thresh] = 1

    stats = get_stats()
    return add_stats(stats, preds, y_hat, y)

def test_all_cc():
    y,cc = get_cc_data()
    for fname in glob.glob('ocsvm_out/*'):
        params = fname.split('/')[-1].split('_')
        print(*params)
        test_one_cc(fname, y,cc)
        print()

if __name__ == '__main__':
    sys.stdout = open('results.txt', 'w+')
    test_all_cc()
    sys.stdout.close()