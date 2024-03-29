from collections import defaultdict
import json 
import socket 

import torch 
from torch_geometric import nn as geo_nn
from torch_geometric.utils import dense_to_sparse, add_remaining_self_loops
from torch_scatter import scatter 

# Depending on which machine we're running on 
if socket.gethostname() == 'colonial0':
    HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'

# Note, this is running over sshfs so it may be slower to load
# may be worth it to make a local copy? 
elif socket.gethostname() == 'orion.ece.seas.gwu.edu':
    HOME = '/home/isaiah/code/NestedGraphs/'

def propagate_labels(g, day, label_f=HOME+'inputs/maybe_mal.txt'):
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
        domain = domain.union(set([int(v[i]['nid']) for i in range(len(v))]))

    for d in domain:
        labels[d] = 1

    # Only consider direct parent-child relations for labeling
    dst = g.edge_index 
    src = get_src(dst, g.csr_ptr).to(dst.device)

    ei = torch.stack([src,dst], dim=0).long()
    ei = ei[:, g.edge_attr[:,0]==1]

    while domain:
        mal = domain.pop()
        children = ei[1][ei[0] == mal]

        for c in children:
            if labels[c] == 0:
                domain.add(c.item())
                labels[c] = labels[mal]+1

    #labels[labels != 0] = 1/labels[labels!=0]
    return labels

def connected_components(g):
    '''
    Find connected components in the process tree only
    '''
    parent = torch.arange(g.x.size(0))
    ei = get_edge_index(g)

    # Filter out IPC (way too many)
    ei = ei[:, g.edge_attr[:,0]==1]

    def find(x):
        if x != parent[x]:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x,y):
        parent_x = find(x)
        parent_y = find(y)

        if parent_x != parent_y:
            parent[parent_y] = parent_x

    for src,dst in ei.T: 
        union(src,dst)

    cc = defaultdict(list)
    proc_ids = torch.arange(g.x.size(0))[g.x[:,0]==1]
    
    # Query with original index, but log the 
    # process-only index
    for i,pid in enumerate(proc_ids):
        cc[find(pid).item()].append(i)

    return [torch.tensor(v) for v in cc.values()]


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
    src = torch.zeros(dst.size(), device=dst.device)

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

    return idx 


def threatrace_feature_extraction(g):
    # To quickly send types to neighbors
    ret = torch.zeros((g.x.size(0), g.edge_feat_dim*2), device=g.x.device)

    ei = get_edge_index(g)
    src_e_types = scatter(g.edge_attr, ei[0], dim=0)
    dst_e_types = scatter(g.edge_attr, ei[1], dim=0)

    # Sometimes nodes don't have src or dst edges so matrices will vary 
    # in size. This makes sure all nodes are accounted for, even if later 
    # indexed nodes don't have edges
    ret[torch.arange(src_e_types.size(0)), :g.edge_feat_dim] = src_e_types
    ret[torch.arange(dst_e_types.size(0)), g.edge_feat_dim:] = dst_e_types

    return ret, ei