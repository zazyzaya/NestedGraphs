import sys 
import time
import pickle
from types import SimpleNamespace

import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from models.tgat import TGAT

if len(sys.argv) > 1:
    DAY = int(sys.argv[1])
else:
    DAY = 23

torch.set_num_threads(8)
criterion = BCEWithLogitsLoss()

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'
HYPERPARAMS = SimpleNamespace(
    t2v=64, hidden=512, out=64, 
    heads=8, layers=3,
    lr=0.001, epochs=100
)

def dot(x1,x2):
    return (x1*x2).sum(dim=1)

def step(model, graph, x, chunks=5):
    '''
    Let's have it do inductive LP for now, maybe update this later?
    '''
    chunk_size = graph.edge_attr.size(0) // chunks 
    pos = []
    neg = []

    for i in tqdm(range(1,chunks-1)):
        src,dst = graph.edge_index[:, chunk_size*i : (chunk_size+1)*i]
        last_t = graph.edge_ts[chunk_size*i]
        
        z = model(
            graph, x, last_t, 
            batch=graph.edge_index[:,:(chunk_size+1)*i].max()
        )

        pos.append(dot(z[src],z[dst]))
        neg.append(dot(
            z[torch.randint(src.max(),(src.size(0),))],
            z[torch.randint(dst.max(),(dst.size(0),))]
        ))

    pos = torch.cat(pos); neg = torch.cat(neg)
    labels = torch.zeros(pos.size(0)+neg.size(0))
    labels[pos.size(0):] = 1. 

    return criterion(torch.cat([pos,neg]), labels)

def train(hp, train_graphs):
    graphs = []
    for t in train_graphs:
        with open(HOME+'inputs/Sept%d/benign/graph%d.pkl' % (DAY, t), 'rb') as f:
            graphs.append(pickle.load(f))
        
        '''
        Cheating to use node embeddings bc they have info about future events
        xs.append(
            torch.load(HOME+'inputs/Sept%d/benign/emb%d.pkl' % (DAY, t))
        )
        '''


    model = TGAT(
        graphs[0].x.size(1), 
        hp.t2v, hp.hidden, hp.out, 
        hp.layers, hp.heads
    )
    opt = Adam(model.parameters(), lr=hp.lr)

    for e in range(hp.epochs):
        for i in range(len(graphs)):
            st = time.time() 

            opt.zero_grad()
            loss = step(model, graphs[i], graphs[i].x)
            loss.backward() 
            opt.step() 

            print("[%d-%d] Loss: %0.4f  (%0.2fs)" % (e,i,loss.item(), time.time()-st))

            torch.save((model.state_dict(), model.args), 'saved_models/embedder/tgat.pkl')



if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    train(HYPERPARAMS, range(1,25))