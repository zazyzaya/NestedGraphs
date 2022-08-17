from turtle import forward
import torch 
from torch import nn 
from torch.nn.utils.rnn import pad_packed_sequence
from torch_geometric.nn import GCNConv, Sequential as GSequential

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

class Time2Vec2d(nn.Module):
    '''
    Recreating Time2Vec: Learning a Vector Representation of Time
        https://arxiv.org/abs/1907.05321
    
    Works on Bx1 matrices (not packed sequences)
    '''
    def __init__(self, dim, is_sin=True):
        super().__init__()
        assert dim > 1, \
            'Must have at least 1 periodic feature, and 1 linear (dim must be >= 2)'
        
        self.lin = nn.Linear(1, dim)
        self.f = torch.sin if is_sin else torch.cos

    def forward(self, times):
        x = self.lin(times)
        periodic = self.f(x[:, 1:])
        return torch.cat([x[:, 0].unsqueeze(-1), periodic], dim=1)
        

class BuiltinAttention(nn.Module):
    '''
    Try using builtin torch nn.Multihead attention function 
    Because some samples have so many values, iterate through them 
    as a list and only do one at a time, otherwise will quickly run out
    of memory 
    '''
    def __init__(self, in_feats, hidden, t_hidden, out, heads=8, layers=4, mean=False):
        super().__init__() 

        self.layers = layers
        self.in_proj = nn.Sequential(
            nn.Linear(in_feats, hidden), 
            nn.ReLU()
        )

        enc = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=t_hidden, dropout=0.1, activation=nn.GELU()
        )
        norm = nn.LayerNorm(hidden)
        self.trans = nn.TransformerEncoder(enc, layers, norm)

        self.project = nn.Linear(hidden, out)
        self.cls = nn.parameter.Parameter(
            torch.rand(1,1,hidden)
        )
        self.sep = nn.parameter.Parameter(
            torch.rand(1,1,hidden)
        )

        self.mean = mean

    def forward(self, ts, x):
        x = packed_cat([ts, x])

        # L x N x d
        x,seq_len = pad_packed_sequence(x)
        x = self.in_proj(x)

        # Add special token 
        x = torch.cat([
            self.cls.repeat(1,x.size(1),1),
            x,
            self.sep.repeat(1,x.size(1),1)
        ], dim=0)
        
        # N x L
        mask = torch.zeros((x.size(1), x.size(0)))

        # Mask out padding (taking care to acct for special token)
        for i in range(x.size(1)):
            mask[i, (seq_len[i]+2):] = 1

        x = self.trans(x, src_key_padding_mask=mask)

        # Get result of special token or mean
        if self.mean: 
            outs = x.sum(dim=1) / seq_len 
        else:
            outs = x[0,:,:]

        return self.project(outs)


class NodeEmbedderSelfAttention(nn.Module):
    def __init__(self, in_feats, hidden, t_hidden, out, heads, layers,
                t2v_dim=8, mean=False):

        super().__init__()

        self.args=(in_feats, hidden, t_hidden, out, heads, layers)
        self.kwargs=dict(t2v_dim=t2v_dim, mean=mean)

        self.t2v = Time2Vec(t2v_dim)
        self.attn = BuiltinAttention(in_feats+t2v_dim, hidden, t_hidden, out, heads=heads, layers=layers)

    def forward(self, ts,x, *args, **kwargs):
        return self.attn(self.t2v(ts), x)

class NodeDecoder(nn.Module):
    def __init__(self, cls_feats, token_feats, hidden, layers, t2v_dim=8):
        super().__init__()

        self.t2v = Time2Vec(t2v_dim)
        self.net = nn.Sequential(
            nn.Linear(cls_feats+token_feats+t2v_dim, hidden), 
            nn.ReLU(), 
            *[
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU()
                )
            for _ in range(layers-1)], 
            nn.Linear(hidden, 1)
        )

    def forward(self, cls, t, x):
        '''
        Given Bxd cls tokens, and LxBxd' sequences of tokens
        return logit of likelihood each element was in that sample 
        '''
        tokens = packed_cat([self.t2v(t), x])
        tokens = pad_packed_sequence(tokens)[0]

        cls = cls.repeat(tokens.size(0),1,1)
        x = torch.cat([cls,tokens], dim=-1)
        return self.net(x)