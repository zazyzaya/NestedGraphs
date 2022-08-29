import glob 
import pickle
import sys

import torch 
from tqdm import tqdm 

from models.tgat import TGAT 
from sim_cse import sample_all

strip_gid = lambda x : x.split('/')[-1].split('.')[0][5:]
HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'

sd,args = torch.load(HOME+'saved_models/embedder/tgat_enc.pkl')
model = TGAT(*args)
model.load_state_dict(sd)
model.eval()

DAY = 23# int(sys.argv[1])
torch.set_num_threads(8)

with torch.no_grad():
    print("Embedding benign hosts")
    prog = tqdm(glob.glob(HOME+'inputs/Sept%d/benign/full_graph*.pkl' % DAY))
    for fname in prog:
        gid = strip_gid(fname)
        prog.desc = gid 

        with open(fname, 'rb') as f:
            graph = pickle.load(f)

        zs = model(graph,graph.x,0,graph.edge_ts.max())
        torch.save(zs, 'inputs/Sept%d/benign/tgat_emb%s.pkl' % (DAY,gid))

    prog.close() 
    

    print("Embedding malicious hosts")
    prog = tqdm(glob.glob(HOME+'inputs/Sept%d/mal/full_graph*.pkl' % DAY))
    for fname in prog:
        gid = strip_gid(fname)
        prog.desc = gid 

        with open(fname, 'rb') as f:
            graph = pickle.load(f)

        with torch.no_grad():
            zs = model(graph,graph.x,0,graph.edge_ts.max())
            torch.save(zs, HOME+'inputs/Sept%d/mal/tgat_emb%s.pkl' % (DAY,gid))

    prog.close()