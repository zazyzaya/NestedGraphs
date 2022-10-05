import torch 
from torch import nn 
from torch_geometric.nn import GATConv, GCNConv

from .tgat import TimeKernel

class GAT(nn.Module):
    def __init__(
            self, in_feats, edge_feats, t_feats, hidden, out, layers, heads, 
            dropout=0.25, device=torch.device('cpu')):

        super().__init__()
        self.args = (in_feats, edge_feats, t_feats, hidden, out, layers, heads)
        self.kwargs = dict(dropout=dropout, device=device)
        
        self.drop = nn.Dropout(p=dropout)
        self.nets = nn.ModuleList(
            [GATConv(in_feats, hidden, heads, edge_dim=(t_feats+edge_feats))] + 
            [GATConv(hidden*heads, hidden, heads, edge_dim=(t_feats+edge_feats)) for _ in range(layers-1)]
        )
        self.proj = nn.Linear(hidden*heads, out, device=device)
        
        if t_feats:
            self.time_kernel = TimeKernel(t_feats, device=device)

        # God damn it, PyG. Fix your API. 
        # Why can't I initialize these in the GPU?
        self.nets.to(device)

        self.edge_feats = edge_feats
        self.t_feats = t_feats

    def forward(self, g, ei, drop=0.5, x=None):
        x = x if x is not None else g.x 

        attr = g.edge_attr
        ts = g.edge_ts.unsqueeze(-1)
        
        # Randomly drop some edges
        mask = torch.rand(ts.size(0)) > drop 
        attr = attr[mask]
        ts = ts[mask]
        ei = ei[:, mask]

        if self.edge_feats:
            if self.t_feats: 
                t2v = self.time_kernel(ts)
                edge_attr = torch.cat([t2v,attr], dim=1)
            else:
                edge_attr = attr 
        
        elif self.t_feats:
            edge_attr = self.time_kernel(ts)  
                  
        else:
            edge_attr = None

        for net in self.nets:
            x = self.drop(torch.relu(net(x, ei, edge_attr=edge_attr)))

        return self.proj(x)