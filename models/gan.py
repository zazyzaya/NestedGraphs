import torch 
from torch import nn 
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv, GCNConv

class NodeGenerator(nn.Module):
    def __init__(self, in_feats, rand_feats, hidden_feats, out_feats, activation=nn.RReLU) -> None:
        super().__init__()

        self.rand_feats = rand_feats 

        # Also learn the distribution to draw samples from? Maybe this is dumb
        self.mean = nn.Linear(in_feats, rand_feats)
        self.std = nn.Linear(in_feats, rand_feats)

        self.net = nn.Sequential(
            nn.Linear(rand_feats, hidden_feats),
            nn.RReLU(),
            nn.Linear(hidden_feats, out_feats),
            activation()
        )

    def forward(self, graph):
        x = graph.x 
        rnd = torch.FloatTensor(x.size(0), self.rand_feats).normal_()
        
        mu = self.mean(x)
        std = self.std(x)
        x = rnd.mul(std).add(mu)

        return self.net(x)

class NodeGeneratorTopology(NodeGenerator):
    def __init__(self, in_feats, rand_feats, hidden_feats, out_feats, activation=nn.RReLU) -> None:
        super().__init__(in_feats, rand_feats, hidden_feats, out_feats, activation)

        self.conv1 = GCNConv(out_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, out_feats)

    def forward(self, graph):
        x = super().forward(graph)
        x = torch.rrelu(self.conv1(x, graph.edge_index))
        return torch.rrelu(self.conv2(x, graph.edge_index))


class GATDiscriminator(nn.Module):
    def __init__(self, emb_feats, in_feats, hidden_feats):
        super().__init__()

        self.gat1 = GATConv(in_feats+emb_feats, hidden_feats, heads=3, concat=False)
        self.gat2 = GATConv(hidden_feats, hidden_feats, heads=3)
        self.lin = nn.Linear(hidden_feats*3, 1)


    def forward(self, z, graph):
        x = torch.cat([z,graph.x], dim=1)
        x = torch.rrelu(self.gat1(x, graph.edge_index))
        x = torch.rrelu(self.gat2(x, graph.edge_index))
        
        # Sigmoid applied later. Using BCE loss w Logits
        return self.lin(x)