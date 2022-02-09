from turtle import forward
import torch 
from torch import nn 
from torch.nn.utils.rnn import pad_packed_sequence

from .utils import repack, packed_cat

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

        self.value = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.RReLU()
        )

        self.out = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.RReLU()
        )

    def forward(self,ts,seq):
        x = packed_cat([ts, seq])

        # Don't want to pad yet, as it adds a bunch of uncecessary 
        # mat muls by 0
        v = repack(self.value(x.data), x)
        k = repack(self.key(x.data), x)
        q = repack(self.query(x.data), x)

        # We've avoided this as long as we could
        k,batch_sizes = pad_packed_sequence(k, batch_first=True)
        q,_ = pad_packed_sequence(q, batch_first=True)

        attn = torch.softmax(q @ k.T, dim=1)
        v,_ = pad_packed_sequence(v, batch_first=True)
        
