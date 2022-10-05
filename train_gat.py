import glob
import pickle
import random
import socket 
from types import SimpleNamespace 

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score, \
    recall_score, precision_score, f1_score, accuracy_score, RocCurveDisplay

import torch 
from torch.nn import BCEWithLogitsLoss, MSELoss, BCELoss
from torch.optim import Adam
from tqdm import tqdm 

from models.gat import GAT
from utils.graph_utils import get_edge_index, get_similar, threatrace_feature_extraction
from loss_fns import contrastive_loss

P_THREADS = 16 # How many threads each worker gets
DEVICE = 3     # Which GPU (for now just use 1)

'''
Uniform batching
-----------------
J,T,time
1,8,76.84  (note htop shows max 5XX% utilization)
4,2,74.84
8,1,64.93
10,1,62.70
12,1,58.81
14,1,49.72
16,1,50.07

Degree-weighted batching
----------------
J,T,time
8,1,65.83
12,1,52.40
16,1,46.36
'''

DAY = 23 

# Depending on which machine we're running on 
if socket.gethostname() == 'colonial0':
    HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/'

# Note, this is running over sshfs so it may be slower to load
# may be worth it to make a local copy? 
elif socket.gethostname() == 'orion.ece.seas.gwu.edu':
    HOME = '/home/isaiah/code/NestedGraphs/inputs/' 

HOME = HOME + 'Sept%d/benign/' % DAY 

hp = HYPERPARAMS = SimpleNamespace(
    tsize=64, hidden=128, heads=1, 
    emb_size=128, layers=3, 
    epochs=100, lr=0.005
)       

bce = BCELoss()
def threatrace(model, g, *args):
    x,ei = threatrace_feature_extraction(g)
    zs = model(g, ei, x=torch.softmax(x, dim=1))

    y = g.x[:, :3]
    y_hat = torch.softmax(zs, dim=1)

    return bce(y_hat, y)

@torch.no_grad()
def test_one(model, g, procs, ccs):
    x, ei = threatrace_feature_extraction(g)
    zs = model(g, ei, x=torch.softmax(x,dim=1))[procs]
    
    # Likelihood of this being a normal process
    results = torch.softmax(zs,dim=1)[:,0]
    
    # Take the average(?) score of the connected component
    cc_results = torch.zeros((len(ccs), 1), device=zs.device)
    for i in range(len(ccs)):
        cc_results[i] = results[ccs[i]].min()

    return cc_results
    

def test(model, thresh=None):
    preds = []; ys = []
    graphs = glob.glob(HOME.replace('benign', 'mal')+'full_graph*')

    for g_file in graphs:
        with open(g_file, 'rb') as f:
            g = pickle.load(f).to(DEVICE)

        # Doesn't really matter which we load, just getting the labels
        embs = torch.load(
            g_file.replace('full_', 'tgat_emb_gat')
        )
        
        procs = embs['proc_mask'].to(DEVICE)
        node_ys = embs['y']
        ccs = embs['ccs']

        ys.append(torch.stack([node_ys[cc].max() for cc in ccs]))
        preds.append(test_one(model, g, procs, ccs))

    # Higher preds -> more anomalous now
    preds = 1-torch.cat(preds).to('cpu')
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


def mean_shifted_cl(model, g, batch):
    labels = get_similar(g.x[batch], depth=3)
    classes, n_classes = labels.unique(return_counts=True)
    ei = get_edge_index(g)

    z = model(g, ei)[batch]
    z_norm = z / z.pow(2).sum(dim=1, keepdim=True).pow(0.5)
    
    c = z.mean(dim=0, keepdim=True)
    c = c / c.pow(2).sum(dim=1, keepdim=True).pow(0.5)

    theta_x = (z_norm-c) / (z_norm-c).pow(2).sum(dim=1, keepdim=True).pow(0.5)
    losses = []
    for i in range(classes.size(0)):
        if n_classes[i] > 1: 
            losses.append(contrastive_loss(
                theta_x[labels == classes[i]], 
                theta_x[labels != classes[i]], 
                assume_normed=True
            ))

    angular_loss = -z_norm * c
    return torch.stack(losses).mean() + angular_loss.mean()

def train(hp):
    # Sets number of threads used by this worker
    torch.set_num_threads(P_THREADS)
    graphs = glob.glob(HOME+'full_graph*')
    
    print("Loading graph")
    with open(graphs[0],'rb') as f:
        g = pickle.load(f)

    tgat = GAT(
        g.edge_feat_dim*2, 0, 
        hp.tsize, hp.hidden, 3, 
        hp.layers, hp.heads,
        device=DEVICE
    )

    opt = Adam(tgat.parameters(), lr=hp.lr)
    loss = torch.tensor([float('nan')])
    for e in range(hp.epochs):
        random.shuffle(graphs)
        prog = tqdm(
            total=len(graphs), 
            desc='[%d-%d] Loss: %0.4f' % (e,0,loss.item())
        )

        for i,g_file in enumerate(graphs):
            with open(g_file,'rb') as f:
                g = pickle.load(f)

            g = g.to(DEVICE)

            # Get this processes batch of jobs. In this case, 
            # nids of nodes that represent processes (x_n = [1,0,0,...,0])
            procs = (g.x[:,0] == 1).nonzero().squeeze(-1)
            opt.zero_grad()
            loss = threatrace(tgat, g, procs)
            loss.backward()
            opt.step() 
            
            prog.desc = '[%d-%d] Loss: %0.4f' % (e,i,loss.item())
            prog.update()

            if i % 25 == 24:
                test(tgat)

            torch.save(
                (
                    tgat.state_dict(), 
                    tgat.args, 
                    tgat.kwargs
                ), 'saved_models/gat.pkl'
            )

        prog.close() 

if __name__ == '__main__':
    train(HYPERPARAMS)