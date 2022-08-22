import torch 
from torch import nn 

class TimeKernel(nn.Module):
    '''
    Nearly identical to Time2Vec but using the TGAT version just
    to be safe
    '''
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, 'TimeKernel must have an even output dimension'
        
        self.w = nn.Linear(1,dim//2)
        self.norm = torch.sqrt(1 / (dim//2))

    def forward(self, t):
        t = t.max()-t
        t = self.w(t)
        return torch.cat([
            torch.sin(t),
            torch.cos(t)
        ], dim=1).true_divide(self.norm)

class QKV(nn.Module):
    def __init__(self, in_feats, hidden):
        self.q = nn.Linear(in_feats, hidden)
        self.k = nn.Linear(in_feats, hidden)
        self.v = nn.Linear(in_feats, hidden)

    def forward(self, z):
        return self.q(z), self.k(z), self.v(z)

class NeighborhoodAggr(nn.Module):
    def __init__(self, in_feats, t_feats, hidden, out_feats):
        super().__init__()

        # Note that W( x||t2v(t) ) = W_1(x) + W_2(t2v)
        # so we can calculate K,Q,V on their own, then add in 
        # the temporal part as if it were part of the initial parameter
        self.t2v = TimeKernel(t_feats)
        self.time_param = nn.Linear(t_feats, in_feats)
        self.proj = nn.Sequential(
            nn.Linear(in_feats+hidden, out_feats),
            nn.ReLU()
        )

    def forward(self, x_0, k,q,v,t, graph):
        # 1xd
        q = q[x_0].unsqueeze(0)

        neighbors,times = graph.one_hop[x_0]
        t_mask = times <= t 

        k,v = k[neighbors[t_mask]], v[neighbors[t_mask]]
        t = self.time_param(
            self.t2v(
                torch.cat(
                    [torch.tensor([[t]]), times], 
                    dim=0
                )
            )
        )

        # Factor time into the original matrices
        q += t[0]; k += t[1:]; v += t[1:]

        # Perform self-attention calculation
        attn = torch.softmax(q@k.T, dim=1)
        h = attn @ v 

        return self.proj(torch.cat([q, ]))


class TGAT(nn.Module):
    def __init__(self, in_feats, hidden, t2v):
        super().__init__()