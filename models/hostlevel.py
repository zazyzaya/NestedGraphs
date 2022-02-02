import torch 
from torch import nn 

from .utils import packed_cat, repack

class Time2Vec(nn.Module):
    '''
    Recreating Time2Vec: Learning a Vector Representation of Time
        https://arxiv.org/abs/1907.05321
    '''
    def __init__(self, dim, is_sin=True):
        super().__init__()
        assert dim > 1, \
            'Must have at least 1 periodic feature, and 1 linear (dim must be >= 2)'
        
        self.lin = nn.Linear(1, dim)
        self.f = torch.sin if is_sin else torch.cos

    def forward(self, times):
        x = times.data 
        x = self.lin(x)
        x[:, 1:] = self.f(x[:, 1:])
        return repack(x, times)


class NodeRNN(nn.Module):
    supported_nets = ['GRU', 'LSTM'] #, 'TRANSFORMER'] TODO

    def __init__(self, in_feats, hidden_feats, out_feats, t2v_feats=8,
                rnn_params={'num_layers': 2}, net='GRU', activation=nn.RReLU) -> None:
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

        
    def forward(self, ts, x, h0=None):
        assert x.data.size(0) == ts.data.size(0), \
            '''ts and x must have the same batch size. 
            Got ts: %s\tx%s''' % (str(ts.data.size()), str(x.data.size()))

        times = self.t2v(ts)
        x = self.rnn(packed_cat([times, x], dim=1), h0)[1]
        return self.linear(x)