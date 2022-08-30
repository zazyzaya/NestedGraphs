import math
from turtle import forward 

from joblib import Parallel, delayed
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
        self.kqv = nn.Linear(in_feats, hidden*3, bias=False)

    def forward(self, z):
        return self.kqv(z).chunk(3,dim=1)

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
    def __init__(self, feats, heads, TKernel, rel_dim=None, dropout=64):
        super().__init__()

        self.kqv = KQV(feats, feats)
        self.neighbor_aggr = NeighborhoodAggr(feats, heads, TKernel, rel_dim, dropout)
        
        self.norm1 = nn.LayerNorm(feats)
        self.norm2 = nn.LayerNorm(feats)

        # Paper uses 2-layer FFNN w/ no final activation and intermediate ReLU
        self.lin = nn.Sequential(
            nn.Linear(feats*2, feats),
            nn.ReLU(),
            nn.Linear(feats, feats)
        )

    USE_MPC = True 
    def forward(self, graph, x, start_t, end_t, batch=0):
        x = self.norm1(x)
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

        if self.USE_MPC:
            futures = [
                torch.jit.fork(
                    self.neighbor_aggr.forward, 
                    nid.item(), 
                    k,q,v, 
                    start_t,end_t, 
                    graph
                )
                for nid in batch
            ]
            h = [torch.jit.wait(f) for f in futures]
            
        else:
            h = []
            for nid in batch:
                h.append(self.neighbor_aggr(nid.item(), k,q,v, start_t,end_t, graph))

        h = torch.cat(h,dim=0)
        # Add in the residual 
        h_2 = h.add(x) 
        h = self.norm2(h_2)

        h = torch.cat([x,h], dim=1)
        h = self.lin(h)

        # Again, add residual from before norm
        return h+h_2


class TGAT(nn.Module):
    def __init__(self, in_feats, edge_feats, t_feats, hidden, out, layers, heads, dropout=64, jit=True):
        super().__init__() 
        self.args = (in_feats, edge_feats, t_feats, hidden, out, layers, heads)
        self.kwargs = dict(dropout=dropout, jit=jit)

        tkernel = TimeKernel(t_feats)
        self.proj = nn.Sequential(
            nn.Linear(in_feats, hidden), 
            nn.ReLU()
        )
        if jit:
            self.layers = nn.ModuleList(
                [JittableTGATLayer(hidden, heads, tkernel, edge_feats, dropout)] * (layers)
            )
        else: 
            self.layers = nn.ModuleList(
                [TGAT_Layer(hidden, heads, tkernel, dropout=dropout, rel_dim=edge_feats)] * (layers)
            )
        self.out_proj = nn.Linear(hidden, out)

    def forward(self, graph, x, start_t, end_t, batch=0):
        # TODO impliment batching 
        x = self.proj(x)
        for layer in self.layers: 
            x = layer(graph, x, start_t, end_t)

        return self.out_proj(x)


class JittableTGATLayer(TGAT_Layer):
    def __init__(self, feats, heads, TKernel, rel_dim, dropout):
        super().__init__(feats, heads, TKernel, rel_dim, dropout)
        
        self.neighbor_aggr = torch.jit.script(JittableSelfAttention_Rels(feats, heads, TKernel, rel_dim))
        self.dropout = dropout
        self.feats = torch.tensor(feats)

    def forward(self, graph, x, start_t, end_t):
        x = self.norm1(x)
    
        batch = torch.arange(x.size(0))
        k,q,v = self.kqv(x)

        futures = []
        for nid in batch:
            nid_ = nid.item()

            # 1xd
            #q_ = q[nid].unsqueeze(0)

            neighbors,times,rels = graph.one_hop[nid_]
            t_mask = (times >= start_t).logical_and(times < end_t)
            t_mask = t_mask.squeeze(-1).nonzero().squeeze(-1)
            
            # Neighborhood dropout
            if t_mask.size(0) > self.dropout and self.training: 
                idx = torch.randperm(t_mask.size(0))
                t_mask = t_mask[idx[:self.dropout]]

            if t_mask.size(0) == 0:
                futures.append(torch.jit.fork(ret_zero, self.feats))
                continue 

            #k_,v_ = k[neighbors[t_mask]], v[neighbors[t_mask]]
            #times,rels = times[t_mask], rels[t_mask]
            
            futures.append(
                torch.jit.fork(
                    self.neighbor_aggr,  
                    k,q,v,
                    neighbors,nid,
                    t_mask, 
                    torch.tensor([[start_t]]),times,rels
                )
            )

        h = [torch.jit.wait(f) for f in futures]
        h = torch.cat(h,dim=0)
        # Add in the residual 
        h_2 = h.add(x) 
        h = self.norm2(h_2)

        h = torch.cat([x,h], dim=1)
        h = self.lin(h)

        # Again, add residual from before norm
        return h+h_2

@torch.jit.script 
def ret_zero(dim):
    return torch.zeros(1,dim)

class JittableSelfAttention_Rels(nn.Module):
    '''
    Essentially the same as NeighborAggr, but must have a 
    deterministic forward function to be compiled for multithreading
    '''
    def __init__(self, hidden,heads,TKernel,r_dim):
        super().__init__()
        self.hidden = hidden 
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm = math.sqrt(1/self.head_dim)

        self.t2v = TKernel
        self.time_params = KQV(TKernel.dim, hidden)
        self.edge_params = KQV(r_dim, hidden)
        
        self.hidden = hidden 
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm = math.sqrt(1/self.head_dim)

    def forward(self, k_,q_,v_, neighbors,nid,mask, start_t,times,rels):
        neigh = neighbors[mask]
        k = k_[neigh]
        q = q_[nid].unsqueeze(0)
        v = v_[neigh]

        times = times[mask]
        rels = rels[mask]

        # Factor time into the original matrices
        t_k, t_q, t_v = self.time_params(
            self.t2v(torch.cat([start_t,times]))
        )
        q = q.add(t_q[0])
        k = k.add(t_k[1:])
        v = v.add(t_v[1:])
 
        r_k, _, r_v = self.edge_params(rels)
        k = k.add(r_k)
        v = v.add(r_v)

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