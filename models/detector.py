import torch 
from torch import nn 
from torch_geometric.nn import GCNConv

class SimpleDetector(nn.Module):
    def __init__(self, in_feats, hidden, layers, device=torch.device('cpu')):
        super().__init__()
        self.args = (in_feats, hidden, layers)
        self.kwargs = dict(device=device)

        self.net = nn.Sequential(
            nn.Linear(in_feats*2, hidden, device=device), 
            nn.Dropout(),
            nn.ReLU(), 
            *[
                nn.Sequential(
                    nn.Linear(hidden, hidden, device=device), 
                    nn.Dropout(),
                    nn.ReLU()
                )
                for _ in range(layers-2)
            ],
            nn.Linear(hidden, 1, device=device)
        )
        self.device = device

    def forward(self, src, dst):
        x = torch.cat([src,dst],dim=1)
        return self.net(x)

    def predict(self, g, zs, procs):
        results = torch.zeros((procs.size(0),1), device=self.device)

        src,dst = [],[]
        idx_map = []; i=0
        for nid in procs:
            st = g.csr_ptr[nid.item()]
            en = g.csr_ptr[nid.item()+1]

            dst.append(g.edge_index[st:en])
            src.append(nid.repeat(en-st))
            idx_map.append(torch.tensor(i, device=self.device).repeat(en-st))
            i += 1

        src = torch.cat(src).long()
        dst = torch.cat(dst).long()
        idx = torch.cat(idx_map).long()
        preds = self.forward(zs[src],zs[dst])

        # Get the neighbor with maximum suspicion to use as the 
        # score for this node
        return results.index_reduce_(0, idx, preds, 'amin', include_self=False)


class GCNDetector(nn.Module):
    def __init__(self, in_dim, hidden, out, layers, device=torch.device('cpu')):
        super().__init__()
        self.args = (in_dim, hidden, layers)
        self.kwargs = dict(device=device)
        
        self.drop = nn.Dropout()
        self.nets = nn.ModuleList(
            [GCNConv(in_dim, hidden)] + 
            [GCNConv(hidden, hidden) for _ in range(layers-2)] +
            [GCNConv(hidden, out)]
        )

        # God damn it, PyG. Fix your API. 
        # Why can't I initialize these in the GPU?
        self.nets.to(device)

    def forward(self, x, ei):
        for net in self.nets[:-1]:
            x = self.drop(torch.relu(net(x, ei)))

        return self.nets[-1](x,ei)

    def predict(self, g, ei, xs, procs):
        zs = self.forward(xs, ei)

        preds = []
        for p in procs.nonzero().squeeze(-1):
            src = zs[p]
            dst = zs[g.get_one_hop(p)[0]]

            edge_likelihoods = src @ dst.T 
            preds.append(edge_likelihoods.min())

        return torch.stack(preds)