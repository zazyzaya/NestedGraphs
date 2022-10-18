import torch
from torch import nn 
from torch_geometric.nn import SAGEConv

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

        self.device = device 
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
        non_leafs = []
        
        # Sample neighbors if available
        for i,d in enumerate(dst_idx_): 
            if d.size(0) == 0:
                continue 

            sample = torch.randint(d.size(0), (self.samples,))
            dst_idx.append(d[sample])
            non_leafs.append(i)

        dst_idx = torch.cat(dst_idx)
        non_leafs = torch.tensor(non_leafs)
        
        # Avoid redundant calculation 
        n_batch, n_idx = dst_idx.unique(return_inverse=True)
        neigh_x = self.forward(
            g, n_batch, layer=layer-1, 
        )[n_idx]

        # Put into groups based on samples (N x S x d)
        neigh_x = neigh_x.reshape(
            non_leafs.size(0), 
            self.samples, 
            neigh_x.size(-1)   
        )

        all_neigh = torch.zeros(
            (batch.size(0), self.samples, neigh_x.size(-1)),
            device=self.device
        )
        all_neigh[non_leafs] = neigh_x
        all_feats = torch.cat([
            src_x.unsqueeze(1).repeat(1,self.samples,1),
            all_neigh
        ], dim=-1)

        # Finally, pass to linear layer
        val = self.layers[layer-1](all_feats)
        
        # For now, just maxpool
        out = val.max(dim=1).values

        # Final layer, project to embedding dim
        if layer == self.n_layers:
            return self.proj_out(out)
        
        # Otherwise
        return torch.relu(out)
        