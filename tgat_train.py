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
mse = torch.nn.MSELoss()
bce = torch.nn.BCEWithLogitsLoss()

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'
HYPERPARAMS = SimpleNamespace(
    t2v=64, hidden=128, out=64, 
    heads=4, layers=3,
    lr=0.001, epochs=100
)

def dot(x1,x2):
    return (x1*x2).sum(dim=1)

def step(enc,dec, opts, graph, x, chunks=5):
    '''
    Encoder/decoder structure
    '''
    enc.train(); dec.train() 

    chunk_size = graph.edge_attr.size(0) // chunks 
    first_t = graph.edge_ts[0]
    prog = tqdm(range(1,chunks))

    for i in prog:
        [o.zero_grad() for o in opts]

        last_t = graph.edge_ts[chunk_size*i]
        
        z = enc(
            graph, x, first_t, last_t
        )
        x_hat = dec(
            graph, z, first_t, last_t
        )

        mask = torch.full((graph.edge_index.size(1),), 1).bool()
        mask[graph.edge_ts < first_t] = False 
        mask[graph.edge_ts > last_t] = False 
        
        # Only check nodes that recieved messages
        # I.e., embs should contain enough info about 
        # a graph's neighbors to reconstruct them
        nodes = graph.edge_index[1,mask].unique()
        print("Running loss on %d nodes" % nodes.size(0))

        mse_loss = mse(
            x[nodes],
            x_hat[nodes]
        )
        topo_loss = -torch.log(
            torch.sigmoid(
                dot(
                    z[graph.edge_index[0,mask]],
                    z[graph.edge_index[1,mask]]
                )
            ) 
        ).mean() 

        loss = mse_loss + topo_loss
        loss.backward() 
        prog.desc = '%0.2f' % loss.item() 
        
        [o.step() for o in opts]
        first_t = last_t

    prog.close()

def train(hp, train_graphs):
    with open(HOME+'inputs/Sept%d/benign/full_graph%d.pkl' % (DAY, train_graphs[0]), 'rb') as f:
        graph = pickle.load(f)

    enc = TGAT(
        graph.x.size(1), 10,
        hp.t2v, hp.hidden, hp.out, 
        hp.layers, hp.heads
    )
    dec = TGAT(
        hp.out, 10,
        hp.t2v, hp.hidden, graph.x.size(1),
        hp.layers, hp.heads
    )
    
    e_opt = Adam(enc.parameters(), lr=hp.lr)
    d_opt = Adam(dec.parameters(), lr=hp.lr)
    opts = [e_opt, d_opt]

    for e in range(hp.epochs):
        for i in train_graphs:
            with open(HOME+'inputs/Sept%d/benign/full_graph%d.pkl' % (DAY, i), 'rb') as f:
                graph = pickle.load(f)

            step(enc, dec, opts, graph, graph.x)
            torch.save((enc.state_dict(), enc.args), 'saved_models/embedder/tgat.pkl')
            torch.save((dec.state_dict(), dec.args), 'saved_models/embedder/tgat.pkl')



if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    train(HYPERPARAMS, range(1,25))