import json 
import socket 

import torch 
from torch_geometric import nn as geo_nn
from torch_geometric.utils import dense_to_sparse, add_remaining_self_loops

# Depending on which machine we're running on 
if socket.gethostname() == 'colonial0':
    HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'

# Note, this is running over sshfs so it may be slower to load
# may be worth it to make a local copy? 
elif socket.gethostname() == 'orion.ece.seas.gwu.edu':
    HOME = '/home/isaiah/code/NestedGraphs/'

def propagate_labels(g, day, label_f=HOME+'inputs/manual_labels.txt'):
    with open(label_f, 'r') as f:
        anoms = json.loads(f.read())

    labels = torch.zeros(g.x.size(0))
    if str(g.gid) not in anoms or str(day) not in anoms[str(g.gid)]:
        return labels 

    # Assume that anomalous processes produce other anomalous processes
    # Luckilly, the graph structure is a tree so we (shouldnt) have to 
    # check for loops and can just do BFS to assign labels 
    domain = set() 
    for v in anoms[str(g.gid)][str(day)].values():
        domain.add(int(v['nid']))

    for d in domain:
        labels[d] = 1

    # Only consider direct parent-child relations for labeling
    dst = g.edge_index 
    src = get_src(dst, g.csr_ptr)

    ei = torch.stack([src,dst], dim=0).long()
    ei = ei[:, g.edge_attr[:,0]==0]

    while domain:
        mal = domain.pop()
        children = ei[1][ei[0] == mal]

        for c in children:
            if labels[c] == 0:
                domain.add(c.item())
                labels[c] = labels[mal]+1

    #labels[labels != 0] = 1/labels[labels!=0]
    return labels


def compress_ei(g):
    '''
    Convert from multi-edge to weighted edge
    This will of course ruin any edge feature vectors
    '''
    mp = geo_nn.MessagePassing(aggr='add')
    x = torch.eye(g.num_nodes)
    x = mp.propagate(
        add_remaining_self_loops(g.edge_index)[0], 
        x=x, size=None
    )
    
    # Produces a weighted adjacency matrix
    ei,ew = dense_to_sparse(x.clamp(0,1))
    g.edge_index = ei
    g.edge_weight = ew

def only_type_edges(g, etype):
    ei = g.edge_index 
    ei = ei[:,g.edge_attr==etype]
    g.edge_index = ei 


def get_src(dst, csr_ptr):
    '''
    Given csr representation of edges, return non-sparse source vector
    '''
    src = torch.zeros(dst.size())

    for i in range(len(csr_ptr)-1):
        src[csr_ptr[i] : csr_ptr[i+1]] = i 

    return src 

def get_edge_index(g):
    src = get_src(g.edge_index, g.csr_ptr)
    return torch.stack([src, g.edge_index]).long()

def update_ptr(src):
    '''
    Given the (sorted) source list, return the compressed
    version in CSR format 
    '''
    idx,cnt = src.unique(return_counts=True)
    ptr = torch.zeros((idx.max()+2,), dtype=torch.long)
    
    last = -1
    offset = 0
    for i in range(idx.size(0)):
        # Node i had neighbors
        if last+1 != idx[i]:
            no_neighbors = idx[i]-last-1
            for j in range(no_neighbors):
                ptr[i+1+offset+j] = ptr[i+offset]
            
            offset += no_neighbors
        
        ptr[i+1+offset] = ptr[i+offset]+cnt[i]
        last = idx[i]
    
    return ptr 


def reindex(batch, ei):
    bmin = batch.min()
    batch = batch-bmin 
    ei = ei-bmin 

    id_map = torch.zeros(batch.max()+1, dtype=torch.long)
    for i,b in enumerate(batch):
        id_map[b] = i 

    
def get_similar(x, feat_dim=3, path_dims=8, depth=1):
    '''
    Given matrix X where the first `feat_dim` columns are a 
    one-hot repr of the node type (proc, file, reg), and the remaining
    columns represent the path (path_dims per layer degrees), return
    the classes of nodes present in the batch, and their indices
    '''

    truncated = x[:,:(feat_dim+path_dims*depth)]
    vals,idx = truncated.unique(dim=0,return_inverse=True)
    n_classes = vals.size(0)

    return idx 
