import glob
import random 
import pickle 
from types import SimpleNamespace 

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score, \
    recall_score, precision_score, f1_score, accuracy_score, RocCurveDisplay
import torch 
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm 

from models.sage import GraphSAGE
from preprocessing.build_unfiltered import ETYPES, NTYPES

DAY = 23
HOME = 'inputs/Sept%d/unfiltered/' % DAY
LABELS = 'inputs/maybe_mal.txt'
PROCS = NTYPES['PROCESS']

DEVICE = 3
HYPERPARAMS = SimpleNamespace(
    hidden=128, layers=2, 
    samples=128, minibatch=25, batch_size=15000, 
    lr=0.001, epochs=1000
)

@torch.no_grad()
def test_one(model, gid, mal_dict):
    g = torch.load(HOME+'mal/%s_unfiltered.pkl' % gid)
    if type(g.ntype) == list:
        g.ntype = torch.tensor(g.ntype)

    g = g.to(DEVICE)
    procs = (g.ntype==PROCS).nonzero().squeeze(-1)
    zs = torch.softmax(model(g, procs), dim=1)

    y_hat = (zs.max(dim=1).indices == PROCS).long() 
    preds = 1-zs[:,PROCS]

    # Get the actual labels
    labels = torch.zeros(g.x.size(0))
    mal = []
    
    for _, uuid_list in mal_dict.items():
        for uuid in uuid_list:
            uuid = uuid['uuid'] # Very readable code, yes
            
            nid = g.node_map.get(uuid)
            if nid:
                mal.append(nid)

    mal = torch.tensor(mal)
    if mal.size(0):
        labels[mal] = 1
    labels = labels[procs]

    return y_hat, preds, labels 


def test(model):
    with open(LABELS, 'r') as f:
        labels = eval(f.read())

    gids = list(labels.keys())
    model.eval()

    y_hat, preds, ys = [],[],[]
    for gid in tqdm(gids, desc='testing'):
        y_h,p,y = test_one(model,gid, labels[gid][str(DAY)])

        y_hat.append(y_h) 
        preds.append(p)
        ys.append(y)

    y_hat = torch.cat(y_hat, dim=0).cpu()
    preds = torch.cat(preds, dim=0).cpu()
    ys = torch.cat(ys, dim=0).cpu()

    stats = dict() 
    stats['Pr'] = precision_score(ys, y_hat, zero_division=1.)
    stats['Re'] = recall_score(ys, y_hat, zero_division=0.)
    stats['F1'] = f1_score(ys, y_hat)
    stats['Ac'] = accuracy_score(ys, y_hat)
    stats['AUC'] = auc_score(ys, preds)
    stats['AP'] = ap_score(ys, preds)

    print("%d samples; %d anomalies" % (ys.size(0), ys.sum().item()))
    for k,v in stats.items():
        print(k,v)

    RocCurveDisplay.from_predictions(ys, preds)
    plt.savefig('roc_curve.png')
    return stats 


ce_loss = CrossEntropyLoss()
def step(g, model, batch_size):
    batch = torch.randperm(g.x.size(0))[:batch_size]
    ys = g.ntype[batch]
    zs = model(g, batch)
    return ce_loss(zs, ys)

def train(hp):
    model = GraphSAGE(
        len(ETYPES)*2, hp.hidden, len(NTYPES),
        hp.layers, samples=hp.samples, 
        device=DEVICE
    )

    opt = Adam(model.parameters(), lr=hp.lr)
    graphs = glob.glob(HOME+'benign/*')
    with open('tt_log.txt', 'w+') as f:
        f.write(str(hp) + '\n\n')

    for e in range(hp.epochs):     
        g = torch.load(random.choice(graphs))

        # Fixed in newer objects, but don't want to rebuild
        # all of them just bc of this
        if type(g.ntype) == list:
            g.ntype = torch.tensor(g.ntype)

        g = g.to(DEVICE)
        prog = tqdm(range(hp.minibatch))
        model.train()
        for _ in prog:
            opt.zero_grad()
            loss = step(g, model, hp.batch_size)
            loss.backward()
            opt.step() 

            prog.desc = '[%d] %0.2f' % (e,loss.item())

        if e % 10 == 9:
            stats = test(model)
            with open('tt_log.txt', 'a') as f:
                f.write('[%d] Loss: %f\n' % (e+1,loss.item()))
                for k,v in stats.items():
                    f.write('%s: %f\n' % (k,v))
                f.write('\n')

        torch.save(
            (model.state_dict(), model.args, model.kwargs), 
            'saved_models/sage.pt'
        )
    
    return model


if __name__ == '__main__':
    train(HYPERPARAMS)