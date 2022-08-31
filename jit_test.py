import pickle
import time
import torch
from types import SimpleNamespace
from models.tgat import TGAT
HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'
hp = SimpleNamespace(
	t2v=64, hidden=2048, out=64, 
	heads=16, layers=3,
	lr=0.001, epochs=100,
    dropout=512
)

with open(HOME+'inputs/Sept23/benign/full_graph1.pkl','rb') as f:
	graph = pickle.load(f)

tgat = TGAT(graph.x.size(1), 10, hp.t2v, hp.hidden, hp.out, hp.layers, hp.heads)

def fwdtime(model,graph):
    # Warm up
    model(graph,torch.rand(graph.x.size()),0,graph.edge_ts.max())

    st = time.time()
    model(graph,graph.x,graph.edge_ts.min(),graph.edge_ts.max())
    return time.time()-st

def bwdtime(model,graph,n_edges=1000):
    print("Fwd...",end='',flush=True)
    st = time.time()
    zs = model(graph,graph.x,graph.edge_ts.min(),torch.tensor([1569244416.0])) # 1000th edge
    print(' (%fs)' % (time.time()-st))

    print("Loss...")
    lp = (zs[graph.edge_index[0,:n_edges]]*zs[graph.edge_index[1,:n_edges]]).sum(dim=1)
    lp = -torch.log(torch.sigmoid(lp)+1e-9)
    loss = lp.mean()

    print("Bwd...")
    st = time.time()
    loss.backward()
    return time.time()-st 

torch.set_num_threads(8)
# DROPOUT=64
#  128 dim hidden layers:
#        8-thr JIT enabled:  26.682656764984130
#       16-thr JIT enabled:  25.477022886276245
#  512 dim hidden layers:
#        8-thr JIT enabled:  26.926122188568115
# 2048 dim hidden layers:
#        8-thr JIT enabled:  38.22869324684143
#
# DROPOUT=512
# 2048 dim hidden layers: 
#        8-thr JIT enabled:  39.88254380226135
tgat = TGAT(graph.x.size(1), 10, hp.t2v, hp.hidden, hp.out, hp.layers, hp.heads, dropout=hp.dropout)
print("JIT enabled: ",bwdtime(tgat,graph))

# DROPOUT=64 (only process 64 at a time while training)
#  128 dim hidden layers:
#        8-thr JIT disabled: 30.47450351715088
#       16-thr JIT disabled: 31.09475350379944
#  512 dim hidden layers:
#        8-thr JIT disabled: 35.43760871887207
# 2048 dim hidden layers: 
#        8-thr JIT disabled: 46.59295392036438
# 
# DROPOUT=512
# 2048 dim hidden layers: 
#        8-thr JIT disabled: 43.10606861114502
tgat = TGAT(graph.x.size(1), 10, hp.t2v, hp.hidden, hp.out, hp.layers, hp.heads, jit=False, dropout=hp.dropout)
print("JIT disabled: ",bwdtime(tgat,graph))