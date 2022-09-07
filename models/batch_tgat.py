import torch 
from torch import nn 

from .attention import MultiHeadAttention
from .tgat import TimeKernel

class BatchTGAT(nn.Module):
    def __init__(self, in_feats, edge_feats, t_feats, hidden, out, layers, heads, neighborhood_size=64, dropout=0.1):
        super().__init__()
        
        self.args = (in_feats, edge_feats, t_feats, hidden, out, layers, heads)
        self.kwargs = dict(neighborhood_size=neighborhood_size, dropout=dropout)

        self.proj_in = nn.Sequential(
            nn.Linear(in_feats, hidden), 
            nn.ReLU()
        )

        self.tkernel = TimeKernel(t_feats)

        d_size = hidden+edge_feats+t_feats

        self.layers = layers
        self.attn_layers = nn.ModuleList(
            [MultiHeadAttention(heads, d_size, d_size, dropout=dropout)] * (layers)
        )
        self.merge_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden+d_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )] * (layers)
        )
        # No output activation in case BCELogits loss is used later
        self.proj_out = nn.Linear(hidden, out)

        # Src nodes don't have an edge feature to pass into 
        # the attention mech. May as well use the free space with
        # information, rather than just sending in zeros? 
        self.src_param = nn.parameter.Parameter(
            torch.rand((1,edge_feats))
        )

        # Sample uniform number of neighbors (at least during training) for 
        # optimized self attention
        self.neighborhood_size = neighborhood_size

    def forward(self, graph, x, start_t=0, end_t=float('inf'), layer=-1, batch=torch.tensor([])):
        if layer==-1:
            layer = self.layers
        
        # h_0 is just node features
        if layer == 0:
            return self.proj_in(x[batch])

        # Generate x for batch nodes
        src_x = self.forward(graph, x, start_t, end_t, layer=layer-1, batch=batch)

        # Then generate x for neighbors
        neighbor_data = [graph.one_hop[b.item()] for b in batch]
        idxs = [(t>start_t).logical_and(t<end_t).squeeze(-1) for _,t,_ in neighbor_data]
        neighbors, ts, rels = [],[],[]
        bs = batch.size(0)
        
        # Get temporal neighbors and their edge data while subsampling if applicable 
        non_leaf_nodes = []
        for i in range(bs):
            idx = idxs[i]

            # Cant process neighbors of nodes that have no neighbors
            if idx.size(0) == 0:
                continue 
            
            non_leaf_nodes.append(i)
            n,t,r = neighbor_data[i]

            # Sample neighborhood with replacement
            # Guarantees tensors of size self.dropout x d are passed to self attn
            idx = idx.nonzero()[
                torch.randint(idx.size(0), (self.neighborhood_size,))
            ].squeeze(-1)
            neighbors.append(n[idx])
            
            # Storing deltas, not just raw times
            t = t[idx]
            ts.append(t.max()-t)
            rels.append(r[idx])
        
        non_leaf_nodes = torch.tensor(non_leaf_nodes)

        # Cat edge features together
        neighbors = torch.cat(neighbors, dim=-1)
        ts = torch.cat(ts, dim=0)
        rels = torch.cat(rels, dim=0)
        
        # Avoid redundant calculation 
        n_batch, n_idx = neighbors.unique(return_inverse=True)
        neigh_x = self.forward(
            graph, x, start_t, end_t, layer=layer-1, 
            batch=n_batch
        )[n_idx]

        # Now cat together edge data with node data
        neigh_x = torch.cat([
            neigh_x, self.tkernel(ts), rels
        ], dim=-1)

        src_x_in = torch.cat([
            src_x, 
            self.tkernel(torch.zeros((1,1))).repeat(bs,1),
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
        full_val = torch.zeros(bs,val.size(-1))
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
