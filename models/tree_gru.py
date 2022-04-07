import torch 
from torch import nn 
from torch_geometric.nn import MessagePassing

class TreeGRUConv(nn.Module):
    '''
    Assuming input is a tree, every node will have exactly 1 or 0
    parents. In a sense, this is a sequence--the walk from parent to 
    child. Following the idea of DAGGNN we try this strategy 
    '''
    def __init__(self, in_feats, hidden, layers, gru_layers=1):
        super().__init__()
        self.gru = nn.GRU(in_feats, hidden, num_layers=gru_layers)
        self.mp = MessagePassing()
        
        self.layers = layers 

    def forward(self, x, ei):
        msg = [x] 
        for i in range(self.layers):
            new_msg = self.mp.propagate(
                ei, x=msg[0], size=None 
            )
            msg = [new_msg] + msg 

        msg = torch.stack(msg)
        return self.gru(msg)[1][-1]
       