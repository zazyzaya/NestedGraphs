from turtle import forward
import torch 
from torch import gru, nn 
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv, GCNConv, GatedGraphConv

from .hostlevel import Time2Vec2d
from .tree_gru import TreeGRUConv

class NodeGenerator(nn.Module):
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


class NodeGeneratorTopology(nn.Module):
    '''
    Replace linear layers w GCNs
    '''
    def __init__(self, in_feats, static_dim, hidden_feats, out_feats, activation=nn.RReLU) -> None:
        super().__init__()
        
        self.conv1 = GCNConv(in_feats+static_dim, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.act = nn.RReLU()
        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(hidden_feats, out_feats)

        self.static_dim = static_dim

    def forward(self, graph):
        x = graph.x 
        if self.static_dim > 0:
            x = torch.cat([
                x, torch.rand(x.size(0), self.static_dim)
            ], dim=1)

        x = self.drop(self.act(self.conv1(x, graph.edge_index)))
        x = self.drop(self.act(self.conv2(x, graph.edge_index)))
        return torch.sigmoid(self.lin(x))


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