from turtle import forward
import torch 
from torch import nn 
from torch.nn.utils.rnn import pad_packed_sequence

from .utils import repack, packed_cat, masked_softmax, get_last_vectors_unpadded

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
        # dot product of each q and V matrix. 
        #
        # Note that the final matrix is (QK^T)V and we only want
        # the last row of it, which is 
        # [(q_n * k_1) * [v_11, v_21, ...] , (q_n * k_2) * [v_12, v_22, ...], etc]
        # Only dependant on the last non-zero row of Q, so why waste the time 
        q = get_last_vectors_unpadded(q)
        attn = q.unsqueeze(1) @ k.transpose(1,2)

        # Removes any values derived from columns further in the future 
        # than exist for this datapoint. Note that zeros only occur when 
        # a padded value influences the weight of an element. We dont even
        # need to know batch sizes, since irrelevant columns are multiplied
        # by the guaranteed zero in masked_softmax
        attn = masked_softmax(attn, 2)
        return (attn @ v).squeeze(1)
        
