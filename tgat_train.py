import sys 
import time
import pickle
from types import SimpleNamespace

import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from models.tgat import TGAT
from models.simple import FFNN

if len(sys.argv) > 1:
    DAY = int(sys.argv[1])
else:
    DAY = 23

torch.set_num_threads(8)
mse = torch.nn.MSELoss()
bce = torch.nn.BCEWithLogitsLoss()

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'
HYPERPARAMS = SimpleNamespace(
	t2v=64, hidden=256, out=128, 
	heads=8, layers=3,
	t_lr=0.0005, d_lr=0.01, epochs=100,
    dropout=20 #Same as original paper
)
def dot(x1,x2):
    return (x1*x2).sum(dim=1)

def step(e,enc,dec, opts, graph, x, chunks=5):
    '''
    Encoder/decoder structure
    '''
    enc.train(); dec.train() 

    chunk_size = graph.edge_attr.size(0) // chunks 
    first_t = graph.edge_ts[0]
    prog = tqdm(total=chunks*2, desc='[%d]'%e)

    for i in range(1,chunks):
        [o.zero_grad() for o in opts]

        last_t = graph.edge_ts[chunk_size*i]
        
        z = enc(
            graph, x, first_t, last_t
        )

        mask = torch.full((graph.edge_index.size(1),), 1).bool()
        mask[graph.edge_ts < first_t] = False 
        mask[graph.edge_ts > last_t] = False 

        # Only check nodes that recieved messages
        # I.e., embs should contain enough info about 
        # a graph's neighbors to reconstruct them
        nodes = graph.edge_index[1,mask].unique()

        x_hat = dec(
            z[nodes]
        )
        mse_loss = mse(
            x[nodes],
            x_hat
        )

        topo_loss_p = torch.sigmoid(
            dot(
                z[graph.edge_index[0,mask]],
                z[graph.edge_index[1,mask]]
            )
        ) 
        topo_loss_n = torch.sigmoid(
            dot(
                z[graph.edge_index[0,mask[torch.randperm(mask.size(0))]]],
                z[graph.edge_index[1,mask[torch.randperm(mask.size(0))]]]
            )
        )

        topo_loss = (-torch.log(1-topo_loss_n+1e-9)-torch.log(topo_loss_p+1e-9)).mean()
        prog.desc = '[%d] R: %0.4f  T: %0.4f' % (e, mse_loss.item(), topo_loss.item())
        prog.update()

        loss = mse_loss+topo_loss
        loss.backward()        
        [o.step() for o in opts]
        prog.update()

        first_t = last_t

    prog.close()

def train(hp, train_graphs):
    with open(HOME+'inputs/Sept%d/benign/full_graph%d.pkl' % (DAY, train_graphs[0]), 'rb') as f:
        graph = pickle.load(f)

    enc = TGAT(
        graph.x.size(1), 10,
        hp.t2v, hp.hidden, hp.out, 
        hp.layers, hp.heads,
        dropout=hp.dropout, jit=False
    )
    dec = FFNN(
        hp.out, hp.hidden, 
        graph.x.size(1), hp.layers
    )
    
    e_opt = Adam(enc.parameters(), lr=hp.t_lr)
    d_opt = Adam(dec.parameters(), lr=hp.d_lr)
    opts = [e_opt, d_opt]

    for e in range(hp.epochs):
        for i in train_graphs:
            with open(HOME+'inputs/Sept%d/benign/full_graph%d.pkl' % (DAY, i), 'rb') as f:
                graph = pickle.load(f)

            step(e, enc, dec, opts, graph, graph.x)
            torch.save((enc.state_dict(), enc.args, enc.kwargs), 'saved_models/embedder/tgat_enc.pkl')
            torch.save((dec.state_dict(), dec.args), 'saved_models/embedder/tgat_dec.pkl')



if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    train(HYPERPARAMS, range(1,25))