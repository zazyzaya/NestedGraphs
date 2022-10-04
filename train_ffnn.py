import glob 
import os 
import pickle 
import random
import socket 
from types import SimpleNamespace

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

from models.detector import SimpleDetector

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
    hidden=128*4, layers=2,
    epochs=50, lr=0.00025
)

bce = nn.BCEWithLogitsLoss()
def step(model, graph, zs, batch, max_edges=2**6): 
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

    sample = torch.randperm(src.size(0))[:max_edges]
    src = src[sample]
    dst = dst[sample]
    neg_d = neg_d[sample]

    pos = model(zs[src], zs[dst])
    neg = model(zs[src], zs[neg_d])

    p_labels = torch.full(pos.size(), 1., device=src.device)
    n_labels = torch.full(neg.size(), 0., device=src.device)

    return bce(
        torch.cat([pos, neg], dim=0),
        torch.cat([p_labels, n_labels], dim=0)
    ), pos.min()


def train(rank, world_size, hp):
    torch.set_num_threads(P_THREADS)
    graphs = glob.glob(HOME+'full_graph*')
    random.shuffle(graphs)

    val_graphs = [graphs.pop() for _ in range(25)]
    model = SimpleDetector(
        128, hp.hidden, hp.layers, device=DEVICE
    )

    opt = Adam(model.parameters(), lr=hp.lr)
    for e in range(hp.epochs):
        random.shuffle(graphs)

        for i,g_file in enumerate(graphs):
            torch.cuda.empty_cache()
            
            with open(g_file,'rb') as f:
                g = pickle.load(f).to(DEVICE)
            
            embs = torch.load(
                g_file.replace('full_', 'tgat_emb_gat')
            )
            zs = embs['zs'].to(DEVICE)
            procs=embs['proc_mask'].to(DEVICE)

            # Get this processes batch of jobs. In this case, 
            # nids of nodes that represent processes (x_n = [1,0,0,...,0])
            bs = procs.size(0) // world_size
            my_batch = procs[bs*rank : bs*(rank+1)]

            model.train()
            opt.zero_grad()
            loss, thresh = step(model, g, zs, my_batch)
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
            if i%25 == 0:
                thresh = float('inf')
                with torch.no_grad():
                    model.eval()
                    for g_file in val_graphs:
                        with open(g_file,'rb') as f:
                            g = pickle.load(f).to(DEVICE)
                        
                        embs = torch.load(
                            g_file.replace('full_', 'tgat_emb_gat')
                        )
                        zs = embs['zs'].to(DEVICE)
                        procs=embs['proc_mask'].to(DEVICE)

                        preds = model.predict(g, zs, procs)
                        thresh = min(thresh, preds.min())
                
                    test_per_cc(model, thresh=thresh)


@torch.no_grad()
def test_one_per_cc(model, g, zs, procs, ccs):
    results = model.predict(g, zs, procs)
    
    # Take the average(?) score of the connected component
    cc_results = torch.zeros((len(ccs), 1), device=zs.device)
    for i in range(len(ccs)):
        cc_results[i] = results[ccs[i]].min()

    return cc_results

def test_per_cc(model, thresh=None):
    preds = []; ys = []
    graphs = glob.glob(HOME.replace('benign', 'mal')+'full_graph*')

    for g_file in graphs:
        with open(g_file, 'rb') as f:
            g = pickle.load(f).to(DEVICE)

        embs = torch.load(
            g_file.replace('full_', 'tgat_emb_gat')
        )
        
        zs = embs['zs'].to(DEVICE)
        procs = embs['proc_mask'].to(DEVICE)
        node_ys = embs['y']
        ccs = embs['ccs']

        ys.append(torch.stack([node_ys[cc].max() for cc in ccs]))
        preds.append(test_one_per_cc(model, g, zs, procs, ccs))

    # Higher preds -> more anomalous now
    preds = 1-torch.sigmoid(torch.cat(preds)).to('cpu')
    ys = torch.cat(ys).to('cpu')

    if thresh is None:
        thresh = preds.quantile(0.999)
    else:
        thresh = 1-torch.sigmoid(thresh).cpu()

    y_hat = torch.zeros(preds.size())
    y_hat[preds > thresh] = 1 

    ys = ys.clamp(0,1)
    
    stats = dict() 
    stats['Pr'] = precision_score(ys, y_hat)
    stats['Re'] = recall_score(ys, y_hat)
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



def main(hp):
    train(0,1,hp)

if __name__ == '__main__':
    main(HYPERPARAMS)