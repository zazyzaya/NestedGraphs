import numpy as np 
import torch 
from torch import nn

'''
Stolen from the TGAT repo: 
    https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs

Good base attention mechanism. Esp for batched input 
'''

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, in_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = hidden_dim
        self.d_v = hidden_dim

        self.w_qs = nn.Linear(in_dim, n_head * hidden_dim, bias=False)
        self.w_ks = nn.Linear(in_dim, n_head * hidden_dim, bias=False)
        self.w_vs = nn.Linear(in_dim, n_head * hidden_dim, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (in_dim + hidden_dim)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (in_dim + hidden_dim)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (in_dim + hidden_dim)))

        self.attention = ScaledDotProductAttention(temperature=np.power(hidden_dim, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(in_dim)

        self.fc = nn.Linear(n_head * hidden_dim, in_dim)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # B x 1 x dk
        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
            
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        return output, attn