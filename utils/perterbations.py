import numpy as np 
import torch 

from .datastructures import FullGraph

'''
Perterbations for graph contrastive learning. Continuing work of 
    https://proceedings.neurips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf
'''
def drop_edge(graph, p=0.1):
    dont_drop = torch.rand(graph.edge_index.size(0)) > p

    dst = graph.edge_index.clone() 
    src = get_src(dst, graph.csr_ptr)

    src = src[dont_drop]
    dst = dst[dont_drop]
    rel = graph.edge_attr[dont_drop].clone()
    ts = graph.edge_ts[dont_drop].clone()
    
    ptr = update_ptr(src)

    return PseudoGraph(
        graph, 
        edge_index=dst, csr_ptr=ptr,
        edge_attr=rel, edge_ts=ts 
    )

def drop_node(graph, p=0.1):
    drop = torch.rand(graph.x.size(0)) <= p

    x = graph.x[~drop].copy() 

    dst = graph.edge_index.clone()
    src = get_src(dst, graph.csr_ptr)

    to_drop = (dst == drop.nonzero()).sum(dim=0) + \
              (src == drop.nonzero()).sum(dim=0)
    idx = ~(to_drop.clamp(0,1).bool())

    src = src[idx]
    dst = dst[idx]
    ptr = update_ptr(src)

    rel = graph.edge_attr[idx].clone()
    ts = graph.edge_ts[idx].clone()
    
    return PseudoGraph(
        graph, x=x,
        edge_index=dst, csr_ptr=ptr,
        edge_attr=rel, edge_ts=ts 
    )

def subgraph(graph, sub_size=0.2): 
    '''
    Following the paper's algorithm, they continue adding
    nodes until the subsample is sub_size * |V| nodes large
    '''
    edges = set()
    nodes = set()

    sub_num = sub_size * graph.num_nodes

    csr = graph.csr_ptr
    dst = graph.edge_index.long()

    seed = np.random.randint(graph.num_nodes, size=1)[0]

    # Make sure initial node has neighbors
    while(csr[seed+1] - csr[seed] == 0):
        seed = np.random.randint(graph.num_nodes, size=1)[0]
    
    nodes.add(seed)
    count = 0
    halt_after = graph.edge_index.size(0) * sub_size
    while(len(nodes) < sub_num):
        # Halting condition, just in case
        # (Also constrains number of edges we can add)
        count += 1 
        if count > halt_after:
            break 

        src = np.random.choice(list(nodes))
        n_idx = torch.arange(csr[src],csr[src+1])
        if n_idx.size(0) == 0:
            continue  

        e_idx = np.random.choice(n_idx)

        edges.add(e_idx)
        nodes.add(dst[e_idx].item())

    nodes = torch.tensor(list(nodes)).sort()[0]
    dst_idx = torch.tensor(list(edges)).sort()[0]

    x = graph.x[nodes].clone() 

    src = get_src(dst, csr)[dst_idx]
    dst = dst[dst_idx].clone()

    # Reindex so all nodes have ids between 0 and |V_s|
    # and match the x matrix after it's indexed (note, nids are sorted
    # so this will work)
    umap = torch.zeros(dst.max()+1, dtype=torch.long)
    uq = dst.unique()
    for i in range(uq.size(0)): 
        umap[uq[i]] = i 

    src = umap[src.long()]
    dst = umap[dst]
    csr = update_ptr(src)

    rel = graph.edge_attr[dst_idx].clone()
    ts = graph.edge_ts[dst_idx].clone()

    return PseudoGraph(
        graph, x=x, 
        edge_index=dst, csr_ptr=csr,
        edge_attr=rel, edge_ts=ts
    )


def attr_mask(graph, p=0.1):
    x = graph.x.clone() 
    x[torch.rand(x.size(0)) <= p] = torch.zeros(x.size(1))

    return PseudoGraph(graph, x=x)


def get_src(dst, csr_ptr):
    src = torch.zeros(dst.size())

    for i in range(len(csr_ptr)-1):
        src[csr_ptr[i] : csr_ptr[i+1]] = i 

    return src 

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


class PseudoGraph(FullGraph):
    '''
    Create a new graph with updated with perturbations 
    Accepts any new changes (e.g. x, edge_index, etc) and
    applies them to itself
    '''
    def __init__(self, graph, **kwargs):
        # If it was provided, set to the new value, otherwise copy the input graph
        apply = lambda key : setattr(self, key, kwargs.get(key, getattr(graph, key).clone()))
        for k in ['x','edge_index','edge_attr','edge_ts','csr_ptr']:
            apply(k)

        self.ready = True 
