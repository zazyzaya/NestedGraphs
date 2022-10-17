import glob
import torch 
import pickle
from joblib import Parallel, delayed

from preprocessing.hasher import path_to_tensor

HOME = 'inputs/'

def add_modules(gid, day=23, is_mal=False, write=True):
    mal_str = 'mal' if is_mal else 'benign'
    g_file = HOME+'Sept%d/%s/full_graph%d.pkl' % (
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
    fmap,feats = torch.load(HOME+'Sept%d/features/%d.pkl' % (day,gid))
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
DEPTH = 8
def add_proc_name(gid, day=23, is_mal=False, write=True):
    mal_str = 'mal' if is_mal else 'benign'
    g_file = HOME+'Sept%d/%s/full_graph%d.pkl' % (
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

def update_both(gid, day=23, is_mal=False, write=True):
    add_proc_name(gid, day, is_mal, write)
    add_modules(gid, day, is_mal, write)

if __name__ == '__main__':
    benign = glob.glob(HOME+'Sept23/benign/*')
    benign = [
        int(b.split('/')[-1].split('_')[-1].replace('.pkl','').replace('graph',''))
        for b in benign
    ]

    mal = glob.glob(HOME+'Sept23/mal/*')
    mal = [
        int(m.split('/')[-1].split('_')[-1].replace('.pkl','').replace('graph',''))
        for m in mal 
    ]

    Parallel(n_jobs=32, prefer='processes')(
        delayed(update_both)(gid, day=23, is_mal=False, write=True)
        for gid in mal 
    )

    Parallel(n_jobs=32, prefer='processes')(
        delayed(update_both)(gid, day=23, is_mal=False, write=True)
        for gid in benign 
    )
