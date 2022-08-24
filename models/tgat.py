import math 

import torch 
from torch import nn

class TimeKernel(nn.Module):
    '''
    Nearly identical to Time2Vec but using the TGAT version just
    to be safe
    '''
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, 'TimeKernel must have an even output dimension'
        
        self.w = nn.Linear(1,dim//2)
        self.norm = math.sqrt(1 / (dim//2))
        self.dim = dim

    def forward(self, t):
        t = t.max()-t
        t = self.w(t)

        # Get sin/cos to be interleaved
        return torch.stack([
            torch.sin(t),
            torch.cos(t)
        ], dim=2).view(t.size(0),self.dim) * self.norm

class KQV(nn.Module):
    def __init__(self, in_feats, hidden):
        super().__init__()

        self.k = nn.Linear(in_feats, hidden)
        self.q = nn.Linear(in_feats, hidden)
        self.v = nn.Linear(in_feats, hidden)

    def forward(self, z):
        return self.k(z), self.q(z), self.v(z)

class NeighborhoodAggr(nn.Module):
    def __init__(self, hidden, heads, TKernel, rel_dim, dropout):
        super().__init__()
        assert hidden % heads == 0, 'KQV feat dim must be divisible by heads'

        # Note that W( x||t2v(t) ) = W_1(x) + W_2(t2v)
        # so we can calculate K,Q,V on their own, then add in 
        # the temporal part as if it were part of the initial parameter
        self.t2v = TKernel
        self.time_params = KQV(TKernel.dim, hidden)
        
        if rel_dim is not None:
            self.edge_params = KQV(rel_dim, hidden)
            self.rels = True 
        else:
            self.rels = False

        self.hidden = hidden 
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm = math.sqrt(1/self.head_dim)

        self.dropout = dropout

    def forward(self, nid, k_,q_,v_, start_t,end_t, graph):
        '''
        Takes as input: 
            nid: the index of the target node 
            k,q,v: the key, query, value matrices for all nodes
            t: the current time 
            graph: a graph datastructure specified by preprocessing.datastructures.HostGraph
        '''
        # 1xd
        q = q_[nid].unsqueeze(0)

        neighbors,times,rels = graph.one_hop[nid]
        t_mask = (times >= start_t).logical_and(times < end_t)
        t_mask = t_mask.squeeze(-1).nonzero().squeeze(-1)
        
        # Neighborhood dropout
        #drop = torch.rand(t_mask.size())
        #t_mask = t_mask[drop >= self.dropout]
        if t_mask.size(0) > self.dropout and self.training: 
            idx = torch.randperm(t_mask.size(0))
            t_mask = t_mask[idx[:self.dropout]]

        if t_mask.size(0) == 0:
            return torch.zeros(1,self.hidden)

        k,v = k_[neighbors[t_mask]], v_[neighbors[t_mask]]

        t_k, t_q, t_v = self.time_params(
            self.t2v(
                torch.cat(
                    [torch.tensor([[start_t]]), times[t_mask]], 
                    dim=0
                )
            )
        )

        # Only necessary for the first layer
        if self.rels: 
            r_k, _, r_v = self.edge_params(rels[t_mask])
            k = k.add(r_k)
            v = v.add(r_v)

        # Factor time into the original matrices
        q = q.add(t_q[0])
        k = k.add(t_k[1:])
        v = v.add(t_v[1:])

        # Use reshape optimization for multihead attn 
        # heads x 1 x d/heads
        q = q.view(1, self.heads, self.head_dim).transpose(0,1)
        
        # heads x d/heads x batch
        k = k.view(k.size(0), self.heads, self.head_dim)
        k = k.transpose(-1,1).transpose(-1,0)

        # heads x batch x d/heads
        v = v.view(v.size(0),self.heads,self.head_dim).transpose(0,1)

        # Perform self-attention calculation
        # output is heads x 1 x d/heads
        # then, "catted" together as 1 x d 
        attn = torch.softmax((q @ k) * self.norm, dim=1)        
        return (attn @ v).view(1,self.hidden)


class TGAT_Layer(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, heads, TKernel, rel_dim=None, dropout=64):
        super().__init__()

        self.kqv = KQV(in_feats, hidden)
        self.neighbor_aggr = NeighborhoodAggr(hidden, heads, TKernel, rel_dim, dropout)
        
        # Paper uses 2-layer FFNN w/ no final activation and intermediate ReLU
        self.lin = nn.Sequential(
            nn.Linear(hidden+in_feats, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_feats)
        )

    
    def forward(self, graph, x, start_t, end_t, batch=0):
        h = []

        if batch <= 0:
            k,q,v = self.kqv(x)
            batch = torch.arange(graph.num_nodes)
        
        # Nodes are numbered in order as they come in, so 
        # if we are only looking at time t, then there is some 
        # max nid s.t. all nodes > nids happen in the future and
        # don't need to be calculated\
        
        # (Actually, as it stands, the nids are not guaranteed to be
        # in order, so can't do this yet TODO)
        else: 
            k,q,v = self.kqv(x[:batch+1])
            batch = torch.arange(batch)

        # TODO node-level parallelism in this for-loop
        for nid in batch:
            h.append(self.neighbor_aggr(nid.item(), k,q,v, start_t,end_t, graph))

        h = torch.cat(h,dim=0)
        x = torch.cat([x,h], dim=1)
        return self.lin(x)


class TGAT(nn.Module):
    def __init__(self, in_feats, edge_feats, t_feats, hidden, out, layers, heads):
        super().__init__() 
        self.args = (in_feats, edge_feats, t_feats, hidden, out, layers, heads)

        tkernel = TimeKernel(t_feats)
        self.layers = nn.ModuleList(
            [TGAT_Layer(in_feats, hidden, hidden, heads, tkernel, rel_dim=edge_feats)] + 
            ([TGAT_Layer(hidden, hidden, hidden, heads, tkernel, rel_dim=edge_feats)] * (layers-2)) + 
            [TGAT_Layer(hidden, hidden, out, heads, tkernel, rel_dim=edge_feats)]
        )

    def forward(self, graph, x, start_t, end_t, batch=0):
        # TODO impliment batching 
        for layer in self.layers: 
            x = layer(graph, x, start_t, end_t)

        return x 