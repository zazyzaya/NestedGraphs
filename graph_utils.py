import json 

import torch 
from torch_geometric import nn as geo_nn
from torch_geometric.utils import dense_to_sparse, add_remaining_self_loops

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'

def propagate_labels(g, day, label_f=HOME+'inputs/manual_labels.txt'):
    with open(label_f, 'r') as f:
        anoms = json.loads(f.read())

    labels = torch.zeros(g.num_nodes)
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
    ei = g.edge_index 
    ei = ei[:, g.edge_attr==0]

    while domain:
        mal = domain.pop()
        children = ei[1][ei[0] == mal]

        for c in children:
            # Troubling that there are loops in here..
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