import glob
import json
import math 
import os 
import pickle
import random
import socket 
import time
from types import SimpleNamespace 

import torch 
import torch.multiprocessing as mp
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import Adam
from tqdm import tqdm 

from models.tgat import TGAT
from utils.graph_utils import get_similar
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
    tsize=64, hidden=64, heads=16, 
    emb_size=128, layers=3, nsize=64,
    epochs=100, lr=0.005
)       

def self_cl_step(model, graph, batch, tau=0.05): 
    '''
    Uses self-contrastive learning with the dropout that's already there
    to produce embeddings that are as self-similar as possible

    https://arxiv.org/pdf/2104.08821.pdf

              sim(x_i, x'_i)
    L = ----------------------------
         Sum(j=/=i) sim(x_i, x'_j)
    '''

    # Pass the same data through the net twice to get embeddings with
    # different dropout masks 
    a = model(graph, batch=batch)
    b = model(graph, batch=batch)
    
    # Compute cosine sim manually 
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    res = torch.exp(res/tau)

    # Self similarity (should be high) is diagonal
    # Dissimilarity to other samples is what remains
    pos = res.diagonal()
    neg = (res.sum(dim=1)-pos)+1e-9

    return (-torch.log(pos/neg)).mean()

def cl_step(model, g, batch):
    labels = get_similar(g.x[batch], depth=2)
    classes, n_classes = labels.unique(return_counts=True)

    z = model(g, batch=batch)

    losses = []
    for i in range(classes.size(0)):
        if n_classes[i] > 1: 
            losses.append(contrastive_loss(
                z[labels == classes[i]], 
                z[labels != classes[i]]
            ))

    return torch.stack(losses).mean()

def mean_shifted_cl(model, g, batch):
    labels = get_similar(g.x[batch], depth=2)
    classes, n_classes = labels.unique(return_counts=True)

    z = model(g, batch=batch)
    z_norm = z / z.norm(dim=1,keepdim=True)
    c = z_norm.mean(dim=0)

    theta_x = (z_norm-c) / (z_norm-c).norm(dim=1,keepdim=True)  
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

    tgat = TGAT(
        g.x.size(1), g.edge_feat_dim, 
        hp.tsize, hp.hidden, hp.emb_size, 
        hp.layers, hp.heads,
        neighborhood_size=hp.nsize,
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
            loss = cl_step(tgat, g, procs)
            loss.backward()
            opt.step() 
            
            prog.desc = '[%d-%d] Loss: %0.4f' % (e,i,loss.item())
            prog.update()

            torch.save(
                (
                    tgat.state_dict(), 
                    tgat.args, 
                    tgat.kwargs
                ), 'saved_models/tgat.pkl'
            )

        prog.close() 

if __name__ == '__main__':
    train(HYPERPARAMS)