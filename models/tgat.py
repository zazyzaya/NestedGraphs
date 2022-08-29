from email.errors import HeaderMissingRequiredValue
import math 

import torch 
from torch import nn
import torch.functional as F

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
        self.kqv = nn.Linear(in_feats, hidden*3)

    def forward(self, z):
        return self.kqv(z).chunk(3, dim=-1)

class NeighborhoodAggr(nn.Module):
    '''
    Self attention aggregation given list of nodes, and
    the embeddings of their neighbors
    '''
    def __init__(self, hidden, heads) -> None:
        super().__init__()
        
        self.hidden = hidden 
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm = math.sqrt(1/self.head_dim)
        

class NeighborhoodSample(nn.Module):
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

        '''
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
        '''

        # Perform self attention when all q,k,v mats are extracted
        return q,k,v 


class TGAT_Layer(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, heads, TKernel, rel_dim=None, dropout=64):
        super().__init__()

        self.kqv = KQV(in_feats, hidden)
        self.neighbor_sample = NeighborhoodSample(hidden, heads, TKernel, rel_dim, dropout)
        self.neighbor_aggr = NeighborhoodAggr
        
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
            h.append(self.neighbor_aggr(
                self.neighbor_sample(
                    nid.item(), k,q,v, start_t,end_t, graph)
                )
            )
                

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


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    '''
    Stolen from 
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

    I just want the multihead attention function. No linear layers, etc. JUST ATTENTION PLEASE
    '''
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = dim
        self.heads = heads
        self.head_dim = dim // heads

    def forward(self, x, mask=None, return_attention=False):
        '''
        Given x as a B x Seq x d*3 tensor, perform self-attention
        (d*3 because splitting into KQV, but using just single MM)
        '''
        batch_size, seq_length, _ = x.size()

        # Separate Q, K, V from linear output
        qkv = x.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)

        if return_attention:
            return values, attention
        else:
            return values