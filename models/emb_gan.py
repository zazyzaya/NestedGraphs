from turtle import forward
import torch 
from torch import gru, nn 
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv, GCNConv, GatedGraphConv

from .hostlevel import Time2Vec2d
from .tree_gru import TreeGRUConv

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

class NodeGeneratorCorrected(nn.Module):
    '''
    I noticed several mistakes in the variational structure
    of the OG NodeGenerator class. I'm correcting theme here,
    but leaving the old class so it's backward compatible w 
    old saved models
    '''
    def __init__(self, in_feats, _, hidden_feats, out_feats, activation=nn.RReLU) -> None:
        # Blank argument so signature matches other gens
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.Dropout(0.25, inplace=True),
            nn.RReLU(),
            nn.Linear(hidden_feats, hidden_feats),
            nn.Dropout(0.25, inplace=True),
            nn.RReLU()
        )   

        self.mu = nn.Linear(hidden_feats, out_feats)
        self.log_std = nn.Linear(hidden_feats, out_feats)

    def forward(self, graph):
        mu, std = self.get_distros(graph)

        # Reparameterize 
        x = torch.FloatTensor(mu.size()).normal_()
        x = x.mul(std).add(mu)
        return x, mu, std


    def get_distros(self, graph):
        x = self.net(graph.x)
        mu = self.mu(x)
        std = self.log_std(x).exp()

        return mu, std

class NodeGeneratorNonVariational(nn.Module):
    '''
    The previous model is so successful when not reparameterizing, 
    I wonder if we even need to make it variational
    '''
    def __init__(self, in_feats, static_dim, hidden_feats, out_feats, activation=nn.RReLU) -> None:
        # Blank argument so signature matches other gens
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_feats+static_dim, hidden_feats),
            nn.Dropout(0.25, inplace=True),
            nn.RReLU(),
            nn.Linear(hidden_feats, hidden_feats),
            nn.Dropout(0.25, inplace=True),
            nn.RReLU(),
            nn.Linear(hidden_feats, out_feats),
            nn.Sigmoid()
        )   

        self.static_dim = static_dim

    def forward(self, graph):
        x = graph.x 
        if self.static_dim > 0:
            x = torch.cat([
                x, torch.rand(x.size(0), self.static_dim)
            ], dim=1)
        
        return self.net(x)

class NodeGeneratorCovar(NodeGeneratorCorrected):
    def __init__(self, in_feats, _, hidden_feats, out_feats, activation=nn.RReLU) -> None:
        super().__init__(in_feats, _, hidden_feats, out_feats, activation)
        
        self.log_std = nn.Linear(
            hidden_feats, out_feats**2
        )
        self.out_feats = out_feats

    def get_distros(self, graph):
        x = self.net(graph.x)
        mu = self.mu(x) 
        cov = self.log_std(x).reshape(x.size(0), self.out_feats, self.out_feats).exp() 

        return mu, cov 

    def forward(self, graph):
        mu, cov = self.get_distros(graph)
        x = torch.FloatTensor(mu.size(0), 1, mu.size(1)).normal_()
        x = (x @ cov).squeeze()
        return x + mu 


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
    def __init__(self, emb_feats, hidden_feats, heads=8):
        super().__init__()

        self.gat1 = GATConv(emb_feats, hidden_feats, heads=heads)
        self.gat2 = GATConv(hidden_feats*heads, hidden_feats, heads=heads)
        self.lin = nn.Linear(hidden_feats*heads, 1)
        self.drop = nn.Dropout(0.25)


    def forward(self, z, graph):
        #x = torch.cat([z,graph.x], dim=1)
        # Experiments showed the graph node feats are unimportant
        # saves a bit of time, and shrinks the model a touch

        x = self.drop(torch.tanh(self.gat1(z, graph.edge_index)))
        x = self.drop(torch.tanh(self.gat2(x, graph.edge_index)))
        
        # Sigmoid applied later. Using BCE loss w Logits
        return self.lin(x)


class GATDiscriminatorTime(GATDiscriminator):
    def __init__(self, emb_feats, hidden_feats, heads=8, t2v_dim=8):
        super().__init__(emb_feats+t2v_dim, hidden_feats, heads=heads)
        self.t2v = Time2Vec2d(t2v_dim)

    def forward(self, z, graph):
        #x = torch.cat([z,graph.x], dim=1)
        # Experiments showed the graph node feats are unimportant
        # saves a bit of time, and shrinks the model a touch
        z = torch.cat([self.t2v(graph.node_times.unsqueeze(-1)), z], dim=1)
        return super().forward(z, graph)


class GCNDiscriminator(GATDiscriminator):
    def __init__(self, emb_feats, hidden_feats, heads=8):
        super().__init__(emb_feats, hidden_feats, heads)
        self.gat1 = GCNConv(emb_feats, hidden_feats)
        self.gat2 = GCNConv(hidden_feats, hidden_feats)
        self.lin = nn.Linear(hidden_feats, 1)
        

class GatedGraphDiscriminator(nn.Module):
    def __init__(self, emb_feats, hidden_feats, heads=8):
        super().__init__() 

        self.gnn1 = GatedGraphConv(hidden_feats, heads)
        self.gnn2 = GatedGraphConv(hidden_feats, heads)
        self.lin = nn.Linear(hidden_feats, 1)

    def forward(self, z, graph):
        x = torch.tanh(self.gnn1(z, graph.edge_index))
        x = torch.tanh(self.gnn2(x, graph.edge_index))
        return self.lin(x)

class TreeGRUDiscriminator(nn.Module):
    def __init__(self, emb_feats, hidden_feats, depth=3, gru_layers=2, **kws):
        super().__init__() 

        self.gru = TreeGRUConv(emb_feats, hidden_feats, depth, gru_layers=gru_layers)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_feats, 1))

    def forward(self, z, graph):
        x = self.gru(z, graph.edge_index)
        return self.out(x)

class TreeGRUGenerator(nn.Module):
    def __init__(self, in_feats, static_dim, hidden_feats, out_feats):
        super().__init__()
        self.node_gen = NodeGeneratorNonVariational(in_feats, static_dim, hidden_feats, hidden_feats)
        self.topology = TreeGRUConv(hidden_feats, out_feats, 3)

    def forward(self, graph):
        z = self.node_gen(graph)
        return torch.sigmoid(self.topology(z, graph.edge_index))

class FFNNDiscriminator(nn.Module):
    '''
    For ablation study. Doesn't work very well, so we know 
    GNNs add value (Gets stuck around AUC 0.5, AP 0.7, ie random)
    '''
    def __init__(self, emb_feats, hidden_feats, heads=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, 1)
        )
    
    def forward(self, z, _):
        return self.net(z)