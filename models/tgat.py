import torch 

def neighbor_aggr(graph, x, t, max_seq=None):
    ei = graph.ei[:,graph.edge_attr <= t]
    idx,cnts = ei.unique(return_counts=True)

    if max_seq is None: 
        max_seq = cnts.max() 

    aggr = torch.zeros(max_seq,idx.size(0),graph.x.size(0))