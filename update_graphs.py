import glob
import torch 
import pickle

from preprocessing.hasher import path_to_tensor

GRAPH_HOME = 'inputs/'
FEAT_HOME = 'feature_extraction/data/features/'

def add_modules(gid, day=23, is_mal=False, write=True):
    mal_str = 'mal' if is_mal else 'benign'
    g_file = GRAPH_HOME+'Sept%d/%s/full_graph%d.pkl' % (
        day, mal_str, gid
    )

    with open(g_file, 'rb') as f: 
        g = pickle.load(f)
    
    # Get uuids of all processes in the graph
    procs = (g.x[:,0] == 1).nonzero()
    uuids = []
    for p in procs:
        uuids.append(g.get(p)['uuid'])

    # Get features of all uuids that had module loads
    fmap,feats = torch.load(FEAT_HOME+'%d.pkl' % gid)
    fmap = {m:i for i,m in enumerate(fmap)}
    feats = feats.to(g.x.device)

    # Construct new matrix for this data
    src,dst = [],[]
    for i,uuid in enumerate(uuids):
        if idx := fmap.get(uuid):
            dst.append(procs[i])
            src.append(idx)

    src = torch.tensor(src)
    dst = torch.cat(dst,dim=0)
    
    feat_mat = torch.zeros(g.x.size(0),feats.size(1), device=g.x.device)
    feat_mat[dst] = feats[src]
    g.x = torch.cat([g.x, feat_mat], dim=1)

    if write:
        with open(g_file, 'wb+') as f: 
            pickle.dump(g, f)

    return g

'''
This should really be done in the graph building phase, 
but may as well add it in here. 
'''
DEPTH = 16
def add_proc_name(gid, day=23, is_mal=False, write=True):
    mal_str = 'mal' if is_mal else 'benign'
    g_file = GRAPH_HOME+'Sept%d/%s/full_graph%d.pkl' % (
        day, mal_str, gid
    )

    with open(g_file, 'rb') as f: 
        g = pickle.load(f)

    # Get uuids of all processes in the graph
    procs = (g.x[:,0] == 1).nonzero()
    uuids = []
    for p in procs:
        uuids.append(g.get(p)['uuid'])

    has_path = []
    paths = []
    for i,uuid in enumerate(uuids):
        if path := g.human_readable.get(uuid):
            has_path.append(procs[i])
            # Trim of PID before sending it to hasher
            path = path.split(':',1)[-1]
            paths.append(path_to_tensor(path, DEPTH))

    has_path = torch.cat(has_path,dim=0)
    paths = torch.stack(paths)
    g.x[has_path, 3:3+paths.size(1)] = paths

    if write:
        with open(g_file, 'wb+') as f: 
            pickle.dump(g, f)

    return g

add_proc_name(201, write=False, is_mal=True)