import torch
from torch import nn 
from torch_geometric.nn import aggr 

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, out, layers, samples=64, pool='max', device=torch.device('cpu')):
        super().__init__()
        self.args = (in_dim, hidden, out, layers)
        self.kwargs = dict(samples=samples, pool=pool, device=device)

        self.layers = nn.ModuleList(
            [nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_dim*2,hidden, device=device)
            )] + 
            [nn.Sequential(
                nn.Dropout(),
                nn.Linear(hidden*2,hidden, device=device)
            )] * (layers-1)
        )
        self.proj_out = nn.Sequential(
            nn.Dropout(), 
            nn.Linear(hidden, out, device=device)
        )

        if pool == 'max':
            self.aggr = aggr.MaxAggregation()
        elif pool == 'mean' or pool == 'avg':
            self.aggr = aggr.MeanAggregation()

        self.device = device 
        self.hidden = hidden 
        self.n_layers = layers
        self.samples = samples

    def forward(self, g, batch, layer=-1):
        if layer == -1:
            layer = self.n_layers

        if layer == 0:
            return g.x[batch]

        # Generate x for batch nodes
        src_x = self.forward(g, batch, layer=layer-1)

        # Then generate x for neighbors
        dst_idx_ = [g.get_one_hop(i) for i in batch]
        dst_idx = []
        pool_idx = []
        
        # Sample neighbors if available
        for i,d in enumerate(dst_idx_): 
            if d.size(0) == 0:
                continue 

            d = d[torch.randperm(d.size(0))[:self.samples]]
            dst_idx.append(d)
            pool_idx += [i]*d.size(0)

        if len(dst_idx):
            dst_idx = torch.cat(dst_idx)
            pool_idx = torch.tensor(pool_idx, device=self.device)
            
            # Avoid redundant calculation 
            n_batch, n_idx = dst_idx.unique(return_inverse=True)
            neigh_x = self.forward(
                g, n_batch, layer=layer-1, 
            )[n_idx]

            # Aggregate neighbors
            neigh_x = self.aggr(neigh_x, pool_idx, dim_size=batch.size(0))

        else:
            dim = self.layers[layer-1][1].in_features - src_x.size(1)
            neigh_x = torch.zeros((src_x.size(0), dim), device=self.device)

        all_feats = torch.cat([
            src_x,
            neigh_x
        ], dim=-1)

        # Finally, pass to linear layer
        out = self.layers[layer-1](all_feats)

        # Final layer, project to embedding dim
        if layer == self.n_layers:
            return self.proj_out(out)
        
        # Otherwise
        return torch.relu(out)