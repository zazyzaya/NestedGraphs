import torch 
from torch import nn 
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv

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

    def forward(self, x):
        rnd = torch.FloatTensor(x.size(0), self.rand_feats).normal_()
        mu = self.mean(x)
        std = self.std(x)
        x = rnd.mul(std).add(mu)

        return self.net(x)


class GATDescriminator(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super().__init__()

        self.gat1 = GATConv(in_feats, hidden_feats, heads=3, concat=False)
        self.gat2 = GATConv(hidden_feats, hidden_feats, heads=3)
        self.lin = nn.Linear(hidden_feats*3, 1)

    def forward(self, x, ei):
        x = torch.rrelu(self.gat1(x, ei))
        x = torch.rrelu(self.gat2(x, ei))
        return torch.sigmoid(self.lin(x))