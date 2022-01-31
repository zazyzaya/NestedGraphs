import torch 
from torch import nn 

class Time2Vec(nn.Module):
    '''
    Recreating Time2Vec: Learning a Vector Representation of Time
        https://arxiv.org/abs/1907.05321
    '''
    def __init__(self, dim, is_sin=True):
        assert dim > 1, \
            'Must have at least 1 periodic feature, and 1 linear (dim must be >= 2)'
        
        self.lin = nn.Linear(1, dim)
        self.f = torch.sin if is_sin else torch.cos

    def forward(self, times):
        x = self.lin(times)
        x[:, 1:] = self.f(x[:, 1:])
        return x


class NodeRNN(nn.Module):
    supported_nets = ['GRU', 'LSTM'] #, 'TRANSFORMER'] TODO

    def __init__(self, in_feats, hidden_feats, out_feats, t2v_feats,
                rnn_params={'layers': 2}, net='GRU', activation=nn.RReLU) -> None:
        super().__init__()

        assert net.upper() in self.supported_nets, \
            "net parameter must be in %s" % str(self.supported_nets)

        if net.upper() == 'GRU': 
            self.rnn = nn.GRU(in_feats+t2v_feats, hidden_feats, **rnn_params)
        else:
            self.rnn = nn.LSTM(in_feats+t2v_feats, hidden_feats, **rnn_params)

        self.t2v = Time2Vec(t2v_feats)
        self.linear = nn.Sequential(
            nn.Linear(hidden_feats, out_feats),
            activation()
        )

        
    def forward(self, x, ts, h0=None):
        times = self.t2v(ts)
        x = self.rnn(torch.cat([times, x], dim=1), h0)
        return self.linear(x)