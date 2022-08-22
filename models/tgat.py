import math 

import torch 
from torch import nn

from models.embedder import KQV 

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

class QKV(nn.Module):
    def __init__(self, in_feats, hidden):
        self.q = nn.Linear(in_feats, hidden)
        self.k = nn.Linear(in_feats, hidden)
        self.v = nn.Linear(in_feats, hidden)

    def forward(self, z):
        return self.q(z), self.k(z), self.v(z)

class NeighborhoodAggr(nn.Module):
    def __init__(self, hidden, heads, TKernel):
        super().__init__()
        assert hidden % heads == 0, 'KQV feat dim must be divisible by heads'

        # Note that W( x||t2v(t) ) = W_1(x) + W_2(t2v)
        # so we can calculate K,Q,V on their own, then add in 
        # the temporal part as if it were part of the initial parameter
        self.t2v = TKernel
        self.time_params = QKV(TKernel.dim, hidden)

        self.hidden = hidden 
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm = math.sqrt(1/self.head_dim)

    def forward(self, nid,k,q,v,t, graph):
        '''
        Takes as input: 
            nid: the index of the target node 
            k,q,v: the key, query, value matrices for all nodes
            t: the current time 
            graph: a graph datastructure specified by preprocessing.datastructures.HostGraph
        '''
        # 1xd
        q = q[nid].unsqueeze(0)

        neighbors,times = graph.one_hop[nid]
        t_mask = times <= t 

        if t_mask.sum() == 0:
            return torch.zeros(1,self.hidden)

        k,v = k[neighbors[t_mask]], v[neighbors[t_mask]]
        t_q, t_k, t_v = self.time_params(
            self.t2v(
                torch.cat(
                    [torch.tensor([[t]]), times], 
                    dim=0
                )
            )
        )

        # Factor time into the original matrices
        q += t_q[0]; k += t_k[1:]; v += t_v[1:]

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
    def __init__(self, in_feats, hidden, out_feats, heads, TKernel):
        super().__init__()

        self.kqv = KQV(in_feats, hidden)
        self.neighbor_aggr = NeighborhoodAggr(hidden, heads, TKernel)
        
        # Paper uses 2-layer FFNN w/ no final activation and intermediate ReLU
        self.lin = nn.Sequential(
            nn.Linear(hidden+in_feats, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_feats)
        )

    
    def forward(self, graph, x, t):
        h = []
        k,q,v = self.kqv(x)

        # TODO node-level parallelism in this for-loop
        for nid in range(graph.num_nodes):
            h.append(self.neighbor_aggr(nid, k,q,v,t,graph))

        h = torch.cat(h,dim=0)
        x = torch.cat([x,h], dim=1)
        return self.lin(x)

class TGAT(nn.Module):
    def __init__(self, in_feats, t_feats, hidden, out, layers, heads):
        super().__init__() 
        tkernel = TimeKernel(t_feats)

        self.layers = nn.ModuleList(
            [TGAT_Layer(in_feats, hidden, hidden, heads, tkernel)] + 
            ([TGAT_Layer(hidden, hidden, hidden, heads, tkernel)] * layers-2) + 
            [TGAT_Layer(hidden, hidden, out, heads, tkernel)]
        )

    def forward(self, graph, x, t):
        for layer in self.layers: 
            x = layer(graph, x, t)

        return x 