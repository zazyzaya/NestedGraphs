import glob 
import os 
import pickle 
import random
import socket 
from types import SimpleNamespace
from xml.etree.ElementInclude import include

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score, \
    recall_score, precision_score, f1_score, accuracy_score, RocCurveDisplay

import torch 
from torch import nn 
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from utils.graph_utils import propagate_labels

P_THREADS = 1
DAY = 23 
DEVICE = 1

# Depending on which machine we're running on 
if socket.gethostname() == 'colonial0':
    HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/'

# Note, this is running over sshfs so it may be slower to load
# may be worth it to make a local copy? 
elif socket.gethostname() == 'orion.ece.seas.gwu.edu':
    HOME = '/home/isaiah/code/NestedGraphs/inputs/'

HOME = HOME + 'Sept%d/benign/' % DAY 

hp = HYPERPARAMS = SimpleNamespace(
    hidden=512, layers=3,
    epochs=50, lr=0.0001
)

class SimpleDetector(nn.Module):
    def __init__(self, in_feats, hidden, layers, device=torch.device('cpu')):
        super().__init__()
        self.args = (in_feats, hidden, layers)
        self.kwargs = dict(device=device)

        self.net = nn.Sequential(
            nn.Linear(in_feats*2, hidden, device=device), 
            nn.Dropout(),
            nn.ReLU(), 
            *[
                nn.Sequential(
                    nn.Linear(hidden, hidden, device=device), 
                    nn.Dropout(),
                    nn.ReLU()
                )
                for _ in range(layers-2)
            ],
            nn.Linear(hidden, 1, device=device)
        )

    def forward(self, src, dst):
        x = torch.cat([src,dst],dim=1)
        return self.net(x)


bce = nn.BCEWithLogitsLoss()
def step(model, graph, zs, batch): 
    src,dst = [],[]

    for nid in batch:
        st = graph.csr_ptr[nid.item()]
        en = graph.csr_ptr[nid.item()+1]

        dst.append(graph.edge_index[st:en])
        src.append(nid.repeat(en-st))

    src = torch.cat(src).long()
    dst = torch.cat(dst).long()
    neg_d = torch.randint(
        0, graph.x.size(0), 
        dst.size()
    )

    pos = model(zs[src], zs[dst])
    neg = model(zs[src], zs[neg_d])

    p_labels = torch.full(pos.size(), 1., device=src.device)
    n_labels = torch.full(neg.size(), 0., device=src.device)

    return bce.forward(
        torch.cat([pos, neg], dim=0),
        torch.cat([p_labels, n_labels], dim=0)
    )


def train(rank, world_size, hp):
    torch.set_num_threads(P_THREADS)
    graphs = glob.glob(HOME+'full_graph*')

    model = SimpleDetector(
        128, hp.hidden, hp.layers, device=DEVICE
    )

    opt = Adam(model.parameters(), lr=hp.lr)
    for e in range(hp.epochs):
        random.shuffle(graphs)

        for i,g_file in enumerate(graphs):
            with open(g_file,'rb') as f:
                g = pickle.load(f).to(DEVICE)
            
            embs = torch.load(
                g_file.replace('full_', 'tgat_emb_clms')
            )
            zs = embs['zs'].to(DEVICE)
            procs=embs['proc_mask'].to(DEVICE)

            # Get this processes batch of jobs. In this case, 
            # nids of nodes that represent processes (x_n = [1,0,0,...,0])
            bs = procs.size(0) // world_size
            my_batch = procs[bs*rank : bs*(rank+1)]

            model.train()
            opt.zero_grad()
            loss = step(model, g, zs, my_batch)
            loss.backward()
            opt.step() 
            
            if rank==0:
                print('[%d-%d] Loss: %0.4f' % (e,i+1,loss.item()))
                '''
                torch.save(
                    (
                        model.state_dict(), 
                        model.args, 
                        model.kwargs
                    ), 'saved_models/anom.pkl'
                )
                '''

            # Try to save some memory 
            #del g,zs,my_batch
            
        test(model)


@torch.no_grad()
def test_one(model, g, zs, procs):
    results = torch.zeros((procs.size(0),1), device=zs.device)

    src,dst = [],[]
    idx_map = []; i=0
    for nid in procs:
        st = g.csr_ptr[nid.item()]
        en = g.csr_ptr[nid.item()+1]

        dst.append(g.edge_index[st:en])
        src.append(nid.repeat(en-st))
        idx_map.append(torch.tensor(i, device=zs.device).repeat(en-st))
        i += 1

    src = torch.cat(src).long()
    dst = torch.cat(dst).long()
    idx = torch.cat(idx_map).long()
    preds = model(zs[src],zs[dst])

    # Get the neighbor with maximum suspicion to use as the 
    # score for this node
    results=results.index_reduce_(0, idx, preds, 'amin', include_self=False)
    return results

def test(model, thresh=None):
    model.eval()
    preds = []; ys = []
    graphs = glob.glob(HOME.replace('benign', 'mal')+'full_graph*')

    for g_file in graphs:
        with open(g_file, 'rb') as f:
            g = pickle.load(f).to(DEVICE)

        embs = torch.load(
            g_file.replace('full_', 'tgat_emb_clms')
        )
        
        zs = embs['zs'].to(DEVICE)
        procs = embs['proc_mask'].to(DEVICE)
        ys.append(embs['y'].to('cpu'))

        preds.append(test_one(model, g, zs, procs))

    # Higher preds -> more anomalous now
    preds = 1-torch.sigmoid(torch.cat(preds)).to('cpu')
    ys = torch.cat(ys).to('cpu')

    if thresh is None:
        thresh = preds.quantile(0.99)

    y_hat = torch.zeros(preds.size())
    y_hat[preds > thresh] = 1 

    preds = preds.clamp(0,1)
    
    stats = dict() 
    stats['Pr'] = precision_score(ys, y_hat)
    stats['Re'] = recall_score(ys, y_hat)
    stats['F1'] = f1_score(ys, y_hat)
    stats['Ac'] = accuracy_score(ys, y_hat)
    stats['AUC'] = auc_score(ys, preds)
    stats['AP'] = ap_score(ys, preds)

    for k,v in stats.items():
        print(k,v)

    RocCurveDisplay.from_predictions(ys, y_hat)
    plt.savefig('roc_curve.png')
    return stats 

def main(hp):
    train(0,1,hp)

if __name__ == '__main__':
    main(HYPERPARAMS)