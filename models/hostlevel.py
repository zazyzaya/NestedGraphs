import torch 
from torch import nn 
from torch_geometric.nn import GATConv

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
        periodic = self.f(x[:, 1:])
        x = torch.cat([x[:, 0].unsqueeze(-1), periodic], dim=1)
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
            Got ts: %s\tx: %s''' % (str(ts.data.size()), str(x.data.size()))

        times = self.t2v(ts)
        x = self.rnn(packed_cat([times, x], dim=1), h0)[1]
        return self.linear(x)


class NodeEmbedder(nn.Module):
    def __init__(self, f_feats, m_feats, r_feats, hidden, out, embed_size):
        super().__init__() 

        self.f_rnn = NodeRNN(f_feats, hidden, out)
        self.m_rnn = NodeRNN(m_feats, hidden, out)
        self.r_rnn = NodeRNN(r_feats, hidden, out)

        self.combo = nn.Linear(out*3, embed_size)

    def forward(self, f, m, r):
        '''
        Expects 3 tuple arguments in the form of 
        (file times, file features),
        (module times, module features),
        (reg times, reg features)
        '''
        f = self.f_rnn(*f)[-1]
        m = self.m_rnn(*m)[-1]
        r = self.r_rnn(*r)[-1]

        x = torch.cat([f,m,r], dim=1)
        return torch.rrelu(self.combo(x))