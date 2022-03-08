import glob 
import pickle

import torch 
from tqdm import tqdm 

from emb_train import sample 

strip_gid = lambda x : x[19:].split('.')[0]
strip_gid_mal = lambda x : x[16:].split('.')[0]

'''
model = torch.load('saved_models/embedder/emb_final.pt')
model.eval() 

print("Embedding 32 dim benign hosts")
for fname in tqdm(glob.glob('inputs/benign/nodes*.pkl')):
    gid = strip_gid(fname)
    with open(fname, 'rb') as f:
        nodes = pickle.load(f)

    data = sample(nodes)
    with torch.no_grad():
        zs = model(data)
    with open('inputs/benign/emb%s_32.pkl' % gid, 'wb+') as f:
        pickle.dump(zs, f)

print("Embedding 32 dim malicious hosts")
for fname in tqdm(glob.glob('inputs/mal/nodes*.pkl')):
    gid = strip_gid_mal(fname)
    print(gid)
    with open(fname, 'rb') as f:
        nodes = pickle.load(f)

    data = sample(nodes)
    with torch.no_grad():
        zs = model(data)

    with open('inputs/mal/emb%s_32.pkl' % gid, 'wb+') as f:
        pickle.dump(zs, f)
'''
model = torch.load('saved_models/embedder/emb_64.pt')
model.eval() 

print("Embedding 64 dim benign hosts")
for fname in tqdm(glob.glob('inputs/benign/nodes*.pkl')):
    gid = strip_gid(fname)
    with open(fname, 'rb') as f:
        nodes = pickle.load(f)

    data = sample(nodes)
    with torch.no_grad():
        zs = model(data)
    with open('inputs/benign/emb%s_64.pkl' % gid, 'wb+') as f:
        pickle.dump(zs, f)

print("Embedding 64 dim malicious hosts")
for fname in tqdm(glob.glob('inputs/mal/nodes*.pkl')):
    gid = strip_gid_mal(fname)
    print(gid)
    with open(fname, 'rb') as f:
        nodes = pickle.load(f)

    data = sample(nodes)
    with torch.no_grad():
        zs = model(data)

    with open('inputs/mal/emb%s_64.pkl' % gid, 'wb+') as f:
        pickle.dump(zs, f)