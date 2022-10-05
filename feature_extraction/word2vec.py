import glob
import random
from types import SimpleNamespace

import torch 
from torch import nn 
from torch.optim import Adam 

from build_modules import get_mod_map
VOCAB_SIZE = len(get_mod_map())

HYPERPARAMETERS = SimpleNamespace(
    emb_dim=128, hidden=512, context_size=5,
    epochs=100, lr=0.01
)

torch.set_num_threads(8)
DEVICE = torch.device('cpu')

class CBOW(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, context_size, device=DEVICE):
        super().__init__()
        self.args = (vocab_size, emb_dim, hidden, context_size)
        self.kwargs = dict(device=device)

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(context_size * emb_dim, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, vocab_size), 
            nn.LogSoftmax(dim=1)
        )
        self.seq_dim = context_size * emb_dim

    def forward(self, seq):
        # Given B x context sequence of indexes, returns B x context x d
        embs = self.emb(seq)

        # Reshape to B x context*emb
        embs = embs.view((seq.size(0), self.seq_dim))
        return self.net(embs)


INPUTS = glob.glob('outputs/modlist_*.pkl')
def sample(context_size, n_targets=1, device=DEVICE):
    '''
    Samples processes' dll sets from a (several?) randomly selected hosts
    Returns contexts (batch x seq_len) and targets (batch x [1,?)])
    '''
    host = torch.load(random.choice(INPUTS))
    contexts = []
    select = context_size + n_targets

    for v in host.values():
        if v.size(0) >= select:
            contexts.append(
                v[torch.randperm(v.size(0))[:select]]
            )

    contexts = torch.stack(contexts).to(device)
    return contexts[:, n_targets:], contexts[:, :n_targets]


def train(hp):
    encoder = CBOW(
        VOCAB_SIZE, hp.emb_dim, hp.hidden, 
        hp.context_size, DEVICE
    )
    opt = Adam(encoder.parameters(), lr=hp.lr)
    criterion = nn.NLLLoss()

    for e in range(hp.epochs):
        opt.zero_grad()
        idx,target = sample(hp.context_size)
        log_probs = encoder(idx)
        loss = criterion(log_probs, target.squeeze(-1))
        loss.backward() 
        opt.step() 

        print('[%d] Loss: %0.4f' % (e,loss.item()))


if __name__ == '__main__':
    train(HYPERPARAMETERS)