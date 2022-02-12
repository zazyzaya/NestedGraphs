import torch 
from torch import nn 
from torch.nn.utils.rnn import pad_packed_sequence

from .utils import packed_cat, packed_fn, repack, packed_cat, packed_aggr, \
    masked_softmax, get_last_vectors_unpadded

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
                rnn_params={'num_layers': 2}, net='LSTM', activation=nn.RReLU) -> None:
        super().__init__()

        assert net.upper() in self.supported_nets, \
            "net parameter must be in %s" % str(self.supported_nets)

        self.net = net.upper()
        if self.net == 'GRU': 
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

        # LSTM returns hidden and cell states; only need hidden
        if self.net == 'LSTM':
            x = x[0]

        return self.linear(x)


class NodeEmbedderRNN(nn.Module):
    def __init__(self, f_feats, r_feats, m_feats, hidden, out, embed_size):
        super().__init__() 

        self.f_rnn = NodeRNN(f_feats, hidden, out)
        self.r_rnn = NodeRNN(r_feats, hidden, out)
        self.m_rnn = NodeRNN(m_feats, hidden, out)

        self.combo = nn.Linear(out*3, embed_size)

    def forward(self, data):
        '''
        Expects 3 tuple arguments in the form of 
        (file times, file features),
        (module times, module features),
        (reg times, reg features)
        '''
        f = self.f_rnn(*data['files'])[-1]
        r = self.r_rnn(*data['regs'])[-1]
        m = self.m_rnn(*data['mods'])[-1]

        x = torch.cat([f,r,m], dim=1)
        return torch.rrelu(self.combo(x))

class KQV_Attention(nn.Module):
    '''
    Takes a packed sequence of tensors as input, computes Key, Query, Value
    attention, and outputs a single vector per list item 
    '''

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()

        self.key = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.RReLU()
        )

        self.query = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.RReLU()
        )
        self.scale = hidden ** (1/2)

        self.value = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.RReLU()
        )

        self.out = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.RReLU()
        )

    def forward(self,ts,seq):
        # Combine time series and features into single packed sequence
        x = packed_cat([ts, seq])

        # Don't want to pad yet, as it adds a bunch of uncecessary 
        # mat muls by 0
        v = repack(self.value(x.data), x)
        k = repack(self.key(x.data), x)
        q = repack(self.query(x.data), x)

        # But its eventually inevitable
        k,_ = pad_packed_sequence(k, batch_first=True)
        v,_ = pad_packed_sequence(v, batch_first=True)

        # As we only care about the final value in the sequence,
        # rather than doing a whole mat mul, we can just take the
        # dot product of the last q vectors and the V matrices. 
        #
        # Note that the final matrix is (QK^T)V and we only want
        # the last row of it, which is 
        # [(q_n * k_1) * [v_11, v_21, ...] , (q_n * k_2) * [v_12, v_22, ...], etc]
        # Only dependant on the last non-zero row of Q, so why waste the time 
        q = get_last_vectors_unpadded(q)
        attn = q.unsqueeze(1) @ k.transpose(1,2)
        attn = attn.true_divide(self.scale)

        # Removes any values derived from columns further in the future 
        # than exist for this datapoint. Note that zeros only occur when 
        # a padded value influences the weight of an element. We dont even
        # need to know batch sizes, since irrelevant columns are multiplied
        # by the guaranteed zero in masked_softmax
        attn = masked_softmax(attn, 2)
        return self.out((attn @ v).squeeze(1))


class NodeEmbedderSelfAttention(nn.Module):
    def __init__(self, f_feats, r_feats, m_feats, hidden, out, embed_size, t2v_dim=8):
        super().__init__()

        self.t2v = Time2Vec(t2v_dim)
        self.f_attn = KQV_Attention(f_feats+t2v_dim, hidden, out)
        self.r_attn = KQV_Attention(r_feats+t2v_dim, hidden, out)
        self.m_attn = KQV_Attention(m_feats+t2v_dim, hidden, out)

        self.combo = nn.Linear(out*3, embed_size)

    def forward(self, data):
        t,f = data['files']
        f = self.f_attn(self.t2v(t), f)

        t,r = data['regs']
        r = self.r_attn(self.t2v(t), r)

        t,m = data['mods']
        m = self.m_attn(self.t2v(t), m)

        x = torch.cat([f,r,m], dim=1)
        return torch.rrelu(self.combo(x))


class DeepAggr(nn.Module):
    def __init__(self, in_feats, hidden, out):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.RReLU(),
            nn.Linear(hidden, out),
            nn.RReLU()
        )

    def forward(self, ts, feats):
        x = packed_cat([ts, feats])
        x = packed_fn(x, self.net)
        return packed_aggr(x)


class NodeEmbedderAggr(nn.Module):
    def __init__(self, f_feats, r_feats, m_feats, hidden, out, embed_size, t2v_dim=8):
        super().__init__()

        self.t2v = Time2Vec(t2v_dim)
        self.f_aggr = DeepAggr(f_feats+t2v_dim, hidden, out)
        self.r_aggr = DeepAggr(r_feats+t2v_dim, hidden, out)
        self.m_aggr = DeepAggr(m_feats+t2v_dim, hidden, out)

        self.combo = nn.Linear(out*3, embed_size)

    def forward(self, data):
        t,f = data['files']
        f = self.f_aggr(self.t2v(t), f)

        t,r = data['regs']
        r = self.r_aggr(self.t2v(t), r)

        t,m = data['mods']
        m = self.m_aggr(self.t2v(t), m)

        x = torch.cat([f,r,m], dim=1)
        return torch.rrelu(self.combo(x))