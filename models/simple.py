import torch 
from torch import nn

class FFNN(nn.Module):
    def __init__(self, in_dim, hidden, out, layers) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(), 
            nn.Dropout(),
            *[
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout()
            ]*(layers-2), 
            nn.Linear(hidden, out), 
            nn.ReLU()
        )
        self.args = (in_dim, hidden, out, layers)

    def forward(self, x):
        return self.net(x)