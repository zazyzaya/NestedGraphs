import torch 
from torch import nn 

from .attention import MultiHeadAttention
from .tgat import TGAT_Layer, TGAT

class BatchTGAT(TGAT):
    def __init__(self, in_feats, edge_feats, t_feats, hidden, out, layers, heads, neighborhood_size=64, dropout=0.1, jit=True):
        super().__init__(in_feats, edge_feats, t_feats, hidden, out, layers, heads, dropout, jit)
        
        self.attn_layers = nn.ModuleList(
            [MultiHeadAttention(heads, hidden, hidden, dropout=dropout)] * (layers)
        )
        self.merge_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden*2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out)
            )] * (layers)
        )

        # Src nodes don't have an edge feature to pass into 
        # the attention mech. May as well use the free space with
        # information, rather than just sending in zeros? 
        self.src_param = nn.parameter.Parameter(
            torch.rand((1,edge_feats))
        )

        # Sample uniform number of neighbors (at least during training) for 
        # optimized self attention
        self.neighborhood_size = neighborhood_size

    def forward(self, graph, x, start_t, end_t, layer=-1, batch=torch.tensor([])):
        if layer==-1:
            layer = len(self.layers)
        
        # h_0 is just node features
        if layer == 0:
            return x[batch]

        # Generate x for batch nodes
        src_x = self.forward(self, graph, x, start_t, end_t, layer=layer-1, batch=batch)

        # Then generate x for neighbors
        neighbor_data = [graph.one_hop[b] for b in batch]
        idxs = [(t>start_t).logical_and(t<end_t) for _,t,_ in neighbor_data]
        neighbors, ts, rels = [],[],[]
        bs = batch.size(0)
        
        # Get temporal neighbors and their edge data while subsampling if applicable
        for i in range(bs):
            idx = idxs[i]
            n,t,r = neighbors[i]

            # Sample neighborhood with replacement
            # Guarantees tensors of size self.dropout x d are passed to self attn
            idx = idx[torch.randint(idx.size(0), (self.neighborhood_size,))]

            neighbors.append(n[idx])
            
            # Storing deltas, not just raw times
            t = t[idx]
            ts.append(t.max()-t)
            
            rels.append(r[idx])
        
        # Cat edge features together
        neighbors = torch.cat(neighbors, dim=-1)
        ts = torch.cat(ts, dim=0)
        rels = torch.cat(rels, dim=0)

        # Generate neighbors' features
        neigh_x = self.forward(
            graph, x, start_t, end_t, layer=layer-1, 
            batch=torch.cat(neighbors, dim=-1)
        )

        # Now cat together edge data with node data
        neigh_x = torch.cat([
            neigh_x, self.tkernel(ts), rels
        ])

        src_x = torch.cat([
            src_x, 
            self.tkernel(torch.zeros((1,1))).repeat(bs,1),
            self.src_param.repeat(bs,1)
        ])

        # B*ns x d -> B x ns x d
        neigh_x = neigh_x.view(bs, self.neighborhood_size, -1)

        # B x d -> B x 1 x d
        src_x = src_x.unsqueeze(1)

        # Finally, use attention mechanism 
        val,attn = self.attn_layers[layer-1](src_x, neigh_x, neigh_x)
