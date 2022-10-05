import glob 
import pickle
import sys

import torch 
from tqdm import tqdm 

from models.tgat import TGAT 
from models.gat import GAT 
from utils.graph_utils import get_edge_index, propagate_labels, connected_components

strip_gid = lambda x : x.split('/')[-1].split('.')[0][5:]
HOME = '/home/isaiah/code/NestedGraphs/'
DEVICE = 1
BATCH_SIZE = 2**14 # About the max that can fit into the GPU

sd,args,kwargs = torch.load(HOME+'saved_models/tgat.pkl')
kwargs['device'] = DEVICE
model = TGAT(*args, **kwargs)
model.load_state_dict(sd)
model.eval()

name = 'tgat'

DAY = 23# int(sys.argv[1])
with torch.no_grad():
    print("Embedding malicious hosts")
    prog = tqdm(glob.glob(HOME+'inputs/Sept%d/mal/full_graph*.pkl' % DAY))
    for fname in prog:
        gid = strip_gid(fname)
        prog.desc = gid 

        with open(fname, 'rb') as f:
            g = pickle.load(f).to(DEVICE)
        
        procs = (g.x[:,0] == 1).nonzero().squeeze(-1)
        
        if type(model) == TGAT:
            print(g.x.size(0))
            zs = torch.cat([
                model(g, batch=torch.arange(i*BATCH_SIZE, min((i+1)*BATCH_SIZE,g.x.size(0))))
                for i in range(g.x.size(0)//BATCH_SIZE + 1)
            ], dim=0)
            print(zs.size(0))
        elif type(model) == GAT: 
            ei = get_edge_index(g)
            zs = model(g, ei, drop=0)

        y = propagate_labels(g, DAY)[procs]
        ccs = connected_components(g)

        torch.save({'zs':zs, 'proc_mask':procs, 'y':y, 'ccs': ccs}, HOME+'inputs/Sept%d/mal/tgat_emb_%s%s.pkl' % (DAY,name,gid))
        del g, zs

    prog.close()


    print("Embedding benign hosts")
    prog = tqdm(glob.glob(HOME+'inputs/Sept%d/benign/full_graph*.pkl' % DAY))
    for fname in prog:
        gid = strip_gid(fname)
        prog.desc = gid 

        with open(fname, 'rb') as f:
            g = pickle.load(f).to(DEVICE)

        procs = (g.x[:,0] == 1).nonzero().squeeze(-1)

        if type(model) == TGAT:
            zs = torch.cat([
                model(g, batch=torch.arange(i*BATCH_SIZE, min((i+1)*BATCH_SIZE,g.x.size(0))))
                for i in range(g.x.size(0)//BATCH_SIZE + 1)
            ], dim=0)
        elif type(model) == GAT: 
            ei = get_edge_index(g)
            zs = model(g, ei, drop=0)

        torch.save({'zs':zs,'proc_mask':procs}, 'inputs/Sept%d/benign/tgat_emb_%s%s.pkl' % (DAY,name,gid))
        del g, zs

    prog.close() 