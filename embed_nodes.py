import glob 
import pickle
import sys

import torch 
from tqdm import tqdm 

from models.embedder import NodeEmbedderSelfAttention, MultiEmbedder
from sim_cse import sample_all

strip_gid = lambda x : x.split('/')[-1].split('.')[0][5:]

# Load file embedder
sd,args,kwargs = torch.load('saved_models/embedder/files_emb.pkl')
f = NodeEmbedderSelfAttention(*args, **kwargs)
f.load_state_dict(sd)

# Load reg embedder
sd,args,kwargs = torch.load('saved_models/embedder/regs_emb.pkl')
r = NodeEmbedderSelfAttention(*args, **kwargs)
r.load_state_dict(sd)

# Wrap them in a single model that cat's their output
model = MultiEmbedder(f,r)

DAY = int(sys.argv[1])

torch.set_num_threads(8)
print("Embedding benign hosts")
for fname in tqdm(glob.glob('inputs/Sept%d/benign/nodes*.pkl' % DAY)):
    gid = strip_gid(fname)
    with open(fname, 'rb') as f:
        nodes = pickle.load(f)

    data = sample_all(nodes)
    with torch.no_grad():
        zs = model(data)
        torch.save(zs, 'inputs/Sept%d/benign/emb%s.pkl' % (DAY,gid))

print("Embedding malicious hosts")
for fname in tqdm(glob.glob('inputs/Sept%d/mal/nodes*.pkl' % DAY)):
    gid = strip_gid(fname)
    #print(gid)
    with open(fname, 'rb') as f:
        nodes = pickle.load(f)

    data = sample_all(nodes)
    with torch.no_grad():
        zs = model(data)
        torch.save(zs, 'inputs/Sept%d/mal/emb%s.pkl' % (DAY,gid))