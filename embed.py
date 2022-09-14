import glob 
import pickle
import sys

import torch 
from tqdm import tqdm 

from models.tgat import TGAT 

strip_gid = lambda x : x.split('/')[-1].split('.')[0][5:]
HOME = '/home/isaiah/code/NestedGraphs/'
DEVICE = 0

sd,args,kwargs = torch.load(HOME+'saved_models/tgat.pkl')
model = TGAT(*args,**kwargs).to(DEVICE)
model.load_state_dict(sd)
model.eval()

DAY = 23# int(sys.argv[1])
with torch.no_grad():
    print("Embedding benign hosts")
    prog = tqdm(glob.glob(HOME+'inputs/Sept%d/benign/full_graph*.pkl' % DAY))
    for fname in prog:
        gid = strip_gid(fname)
        prog.desc = gid 

        with open(fname, 'rb') as f:
            g = pickle.load(f).to(DEVICE)

        procs = (g.x[:,0] == 1).nonzero().squeeze(-1)
        zs = model(g, batch=procs)
        torch.save(zs, 'inputs/Sept%d/benign/tgat_emb%s.pkl' % (DAY,gid))
        del g, zs

    prog.close() 
    

    print("Embedding malicious hosts")
    prog = tqdm(glob.glob(HOME+'inputs/Sept%d/mal/full_graph*.pkl' % DAY))
    for fname in prog:
        gid = strip_gid(fname)
        prog.desc = gid 

        with open(fname, 'rb') as f:
            g = pickle.load(f).to(DEVICE)
        
        procs = (g.x[:,0] == 1).nonzero().squeeze(-1)
        zs = model(g,batch=procs)
        torch.save(zs, HOME+'inputs/Sept%d/mal/tgat_emb%s.pkl' % (DAY,gid))
        del g, zs

    prog.close()