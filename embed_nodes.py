import glob 
import pickle

import torch 
from tqdm import tqdm 

from models.hostlevel import NodeEmbedderSelfAttention
from emb_train import sample 

strip_gid = lambda x : x[19:].split('.')[0]
strip_gid_mal = lambda x : x[16:].split('.')[0]

model = NodeEmbedderSelfAttention(
    37, 35, None, 16, 8, 32, t2v_dim=8
)
model.load_state_dict(torch.load('saved_models/embedder/emb_statedict.pt'))
model.eval() 

print("Embedding benign hosts")
for fname in tqdm(glob.glob('inputs/benign/nodes*.pkl')):
    gid = strip_gid(fname)
    with open(fname, 'rb') as f:
        nodes = pickle.load(f)

    data = sample(nodes)
    with torch.no_grad():
        zs = model(data)
    with open('inputs/benign/emb%s.pkl' % gid, 'wb+') as f:
        pickle.dump(zs, f)

print("Embedding malicious hosts")
for fname in tqdm(glob.glob('inputs/mal/nodes*.pkl')):
    gid = strip_gid_mal(fname)
    print(gid)
    with open(fname, 'rb') as f:
        nodes = pickle.load(f)

    data = sample(nodes)
    with torch.no_grad():
        zs = model(data)

    with open('inputs/mal/emb%s.pkl' % gid, 'wb+') as f:
        pickle.dump(zs, f)