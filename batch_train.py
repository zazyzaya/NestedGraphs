import glob 
import pickle
import time 
from types import SimpleNamespace

import torch 

from models.batch_tgat import BatchTGAT 

DAY = 23 
HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/Sept%d/benign/' % DAY 
hp = HYPERPARAMS = SimpleNamespace(
    tsize=64, hidden=64, heads=16, 
    emb_size=128, layers=3, nsize=64
)

torch.set_num_threads(8)
files = glob.glob(HOME+'full_graph*')
with open(files[0],'rb') as f:
    g = pickle.load(f)

tgat = BatchTGAT(
    g.x.size(1), g.edge_feat_dim, 
    hp.tsize, hp.hidden, hp.emb_size, 
    hp.layers, hp.heads,
    neighborhood_size=hp.nsize
)

# First 100 processes 
batch = (g.x[:,0] == 1).nonzero().squeeze(-1)[:100]

st = time.time()
tgat(g, g.x, batch=batch)
print(time.time()-st)