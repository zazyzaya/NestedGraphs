import math 

import numpy as np 
import torch 
from torch import nn 

class TimeKernel(nn.Module):
    '''
    Nearly identical to Time2Vec but using the TGAT version just
    to be safe
    '''
    def __init__(self, dim, device=torch.device('cpu')):
        super().__init__()
        assert dim % 2 == 0, 'TimeKernel must have an even output dimension'
        
        self.w = nn.Linear(1,dim//2, device=device)
        self.norm = math.sqrt(1 / (dim//2))
        self.dim = dim

    def forward(self, t):
        #t = t.max()-t
        t = self.w(t)

        # Get sin/cos to be interleaved
        return torch.stack([
            torch.sin(t),
            torch.cos(t)
        ], dim=2).view(t.size(0),self.dim) * self.norm

'''
Stolen from the TGAT repo: 
    https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs

Good base attention mechanism. Esp for batched input 
'''
class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, in_dim, hidden_dim, dropout=0.1, device=torch.device('cpu')):
        super().__init__()

        self.n_head = n_head
        self.d_k = hidden_dim
        self.d_v = hidden_dim

        self.w_qs = nn.Linear(in_dim, n_head * hidden_dim, bias=False, device=device)
        self.w_ks = nn.Linear(in_dim, n_head * hidden_dim, bias=False, device=device)
        self.w_vs = nn.Linear(in_dim, n_head * hidden_dim, bias=False, device=device)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (in_dim + hidden_dim)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (in_dim + hidden_dim)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (in_dim + hidden_dim)))

        self.attention = ScaledDotProductAttention(temperature=np.power(hidden_dim, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(in_dim, device=device)

        self.fc = nn.Linear(n_head * hidden_dim, in_dim, device=device)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # B x 1 x dk
        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
            
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        return output, attn

class TGAT(nn.Module):
    '''
    Adapted from the TGAT repo for our data structure
    '''
    def __init__(self, in_feats, edge_feats, t_feats, hidden, out, layers, heads, neighborhood_size=64, dropout=0.1, device=torch.device('cpu')):
        super().__init__()
        
        self.args = (in_feats, edge_feats, t_feats, hidden, out, layers, heads)
        self.kwargs = dict(neighborhood_size=neighborhood_size, dropout=dropout, device=device)

        self.proj_in = nn.Sequential(
            nn.Linear(in_feats, hidden, device=device), 
            nn.ReLU()
        )

        self.tkernel = TimeKernel(t_feats, device=device)

        d_size = hidden+edge_feats+t_feats

        self.layers = layers
        self.attn_layers = nn.ModuleList(
            [MultiHeadAttention(heads, d_size, d_size, dropout=dropout, device=device)] * (layers)
        )
        self.merge_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden+d_size, hidden, device=device),
                nn.ReLU(),
                nn.Linear(hidden, hidden, device=device)
            )] * (layers)
        )
        # No output activation in case BCELogits loss is used later
        self.proj_out = nn.Linear(hidden, out, device=device)

        # Src nodes don't have an edge feature to pass into 
        # the attention mech. May as well use the free space with
        # information, rather than just sending in zeros? 
        self.src_param = nn.parameter.Parameter(
            torch.rand((1,edge_feats), device=device)
        )

        # Sample uniform number of neighbors (at least during training) for 
        # optimized self attention
        self.neighborhood_size = neighborhood_size
        self.device = device 

    def forward(self, graph, start_t=0., end_t=float('inf'), layer=-1, batch=torch.tensor([])):
        if layer==-1:
            layer = self.layers
        
        # h_0 is just node features
        if layer == 0:
            return self.proj_in(graph.x[batch])

        # Generate x for batch nodes
        src_x = self.forward(graph, start_t, end_t, layer=layer-1, batch=batch)

        # Then generate x for neighbors
        neighbor_data = [graph.get_one_hop(b.item()) for b in batch]
        idxs = [
            (t>start_t).logical_and(t<end_t).squeeze(-1) 
            for _,t,_ in neighbor_data 
        ]
        neighbors, ts, rels = [],[],[]
        bs = batch.size(0)
        
        # Get temporal neighbors and their edge data while subsampling if applicable 
        non_leaf_nodes = []
        for i in range(bs):
            idx = idxs[i]
            n,t,r = neighbor_data[i]
            
            # Skip neighborless nodes
            if not n.size(0):
                continue 

            non_leaf_nodes.append(i)

            # For some reason, the list comp above makes nodes with a single neighbor 
            # into 1d tensors. Adding this check seems easier than fixing that issue
            if n.dim() == 1:  
                idx = torch.zeros((self.neighborhood_size,), dtype=torch.long, device=self.device)

            # Sample neighborhood with replacement
            # Guarantees tensors of size self.dropout x d are passed to self attn
            else:
                idx = idx.nonzero()[
                    torch.randint(idx.size(0), (self.neighborhood_size,), device=self.device)
                ].squeeze(-1)
            
            neighbors.append(n[idx])
            
            # Storing deltas, not just raw times
            t = t[idx]
            ts.append(t.max()-t)
            rels.append(r[idx])
        
        non_leaf_nodes = torch.tensor(non_leaf_nodes, device=self.device)

        # Cat edge features together
        neighbors = torch.cat(neighbors, dim=-1)
        ts = torch.cat(ts, dim=0).unsqueeze(-1)
        rels = torch.cat(rels, dim=0)
        
        # Avoid redundant calculation 
        n_batch, n_idx = neighbors.unique(return_inverse=True)
        neigh_x = self.forward(
            graph, start_t, end_t, layer=layer-1, 
            batch=n_batch.long()
        )[n_idx]

        # Now cat together edge data with node data
        neigh_x = torch.cat([
            neigh_x, self.tkernel(ts), rels
        ], dim=-1)

        src_x_in = torch.cat([
            src_x, 
            self.tkernel(torch.zeros((1,1), device=self.device)).repeat(bs,1),
            self.src_param.repeat(bs,1)
        ], dim=-1)[non_leaf_nodes]

        # B*ns x d -> B x ns x d
        neigh_x = neigh_x.view(non_leaf_nodes.size(0), self.neighborhood_size, -1)

        # B x d -> B x 1 x d
        src_x_in = src_x_in.unsqueeze(1)

        # Finally, use attention mechanism 
        val,attn = self.attn_layers[layer-1](src_x_in, neigh_x, neigh_x)
        
        # Put any aggregations of neighbors into matrix, 
        # nodes with no children are left as zero
        full_val = torch.zeros(bs,val.size(-1), device=self.device)
        full_val[non_leaf_nodes] = val.squeeze(1)

        # And like GraphSAGE, cat to the original vector, and FFNN 
        # B x (hidden + hidden + e_feats + t_feats) -> B x hidden
        out = self.merge_layers[layer-1](
            torch.cat(
                [src_x, full_val], 
                dim=1
            )
        )

        # Final layer, project to embedding dim
        if layer == self.layers:
            return self.proj_out(out)
        
        # Otherwise
        return out 


class TGATGraph(nn.Module):
    def __init__(self, in_feats, edge_feats, t_feats, hidden, out, layers, heads, neighborhood_size=64, dropout=0.1, device=torch.device('cpu')):
        super().__init__()
        
        self.tgat = TGAT(in_feats, edge_feats, t_feats, hidden, out, layers, heads, neighborhood_size, dropout, device)
        self.pred_net = nn.Sequential(
            nn.Linear(out, out*2, device=device),
            nn.ReLU(),
            nn.Linear(out*2, out, device=device), 
            nn.ReLU() 
        )

    def embed(self, graph, start_t=0., end_t=float('inf'), batch=torch.tensor([])):
        return self.tgat(graph, start_t, end_t, batch=batch)

    def forward(self, graph, start_t=0., end_t=float('inf'), batch=torch.tensor([])):
        zs = self.embed(graph, start_t, end_t, batch=batch)
        
        # Readout fn is just sum as proposed by SAGE
        zs = zs.sum(dim=0)

        return self.pred_net(zs)