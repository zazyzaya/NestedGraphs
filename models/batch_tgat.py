import torch 
from torch import nn 

from .attention import MultiHeadAttention
from .tgat import TGAT_Layer, TGAT

class BatchTGAT(TGAT):
    def __init__(self, in_feats, edge_feats, t_feats, hidden, out, layers, heads, dropout=64, jit=True):
        super().__init__(in_feats, edge_feats, t_feats, hidden, out, layers, heads, dropout, jit)
        
        self.layers = nn.ModuleList(
            [MultiHeadAttention(hidden, heads, dropout=dropout, rel_dim=edge_feats)] * (layers)
        )
        self.dropout = dropout 

    def forward(self, graph, x, start_t, end_t, layer=-1, batch=torch.tensor([])):
        if layer==-1:
            layer = len(self.layers)
        
        neighbors = [graph.one_hop[b] for b in batch]
        idxs = [(t>start_t).logical_and(t<end_t) for _,t,_ in neighbors]
        nodes, ts, rels, masks = [],[],[],[]
        bs = batch.size(0)
        
        # Get temporal neighbors and their edge data
        # TODO impliment neighborhood subsampling here
        for i in range(bs):
            idx = idxs[i]
            n,t,r = neighbors[i]

            if self.training and idx.size(0) > self.dropout:
                idx = idx[torch.randperm(idx.size(0))[:self.dropout]]
                masks.append(torch.zeros(self.dropout))

            nodes.append(n[idx])
            ts.append(self.tkernel(t[idx]))
            rels.append(r[idx])

        # h_0 is just node features
        if layer == 0:
            x_n = [x[src] for src in nodes]
            return [
                torch.cat(
                    [x_n[i], rels[i], ts[i]], dim=1
                ) for i in range(bs) 
            ]
        
        # Is there a way to avoid this loop and do this all at once? 
        h = [
            self.forward(graph, x, start_t, end_t, layer=layer-1, batch=n)
            for n in nodes
        ]



        