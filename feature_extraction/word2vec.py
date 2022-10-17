import glob
import random
from types import SimpleNamespace

import torch 
from torch import nn 
from torch.optim import Adam 
from tqdm import tqdm 

from build_modules import get_mod_map
VOCAB_SIZE = len(get_mod_map())

HYPERPARAMETERS = SimpleNamespace(
    emb_dim=128, hidden=512, context_size=5, 
    epochs=1000, lr=0.001
)

torch.set_num_threads(8)
DEVICE = 3 #torch.device('cpu')

class CBOW(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, context_size, device=DEVICE):
        super().__init__()
        self.args = (vocab_size, emb_dim, hidden, context_size)
        self.kwargs = dict(device=device)

        self.emb = nn.Embedding(vocab_size, emb_dim, device=device)
        self.net = nn.Sequential(
            nn.Linear(context_size * emb_dim, hidden, device=device),
            nn.ReLU(), 
            nn.Linear(hidden, vocab_size, device=device), 
            nn.LogSoftmax(dim=1)
        )
        self.seq_dim = context_size * emb_dim

    def forward(self, seq):
        # Given B x context sequence of indexes, returns B x context x d
        embs = self.emb(seq)

        # Reshape to B x context*emb
        embs = embs.view((seq.size(0), self.seq_dim))
        return self.net(embs)


INPUTS = glob.glob('data/inputs/modlist_*.pkl')
def sample(context_size, n_targets=1, device=DEVICE, hosts=5):
    '''
    Samples processes' dll sets from a (several?) randomly selected hosts
    Returns contexts (batch x seq_len) and targets (batch x [1,?)])
    '''
    contexts = []
    for _ in range(hosts):
        host = torch.load(random.choice(INPUTS))
        select = context_size + n_targets

        sizes = []
        for v in host.values():
            sizes.append(v.size(0))
            if v.size(0) >= select:
                contexts.append(
                    v[torch.randperm(v.size(0))[:select]]
                )

    #print('Avg len: %f' % (sum(sizes) / len(sizes)), 'min: %d' % min(sizes), 'max: %d' % max(sizes))
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

        if e % 25 == 24:
            torch.save(encoder.emb, 'CBOW_emb.pt')

OUT_DIR = 'data/features/'
@torch.no_grad()
def embed():
    model = torch.load('embedder.pt')
    for i,host_file in enumerate(INPUTS): 
        ids, feats = [],[]
        host = torch.load(host_file)
        
        prog = tqdm(desc='(%d/%d)' % (i+1,len(INPUTS)), total=len(host))
        for k,v in host.items():
            ids.append(k)
            feats.append(
                model(v.to(DEVICE)).mean(dim=0)
            )
            prog.update()

        gid = host_file.split('_')[1].replace('.pkl', '')
        torch.save(
            [ids, torch.stack(feats)], 
            OUT_DIR+gid+'.pkl'
        )

        

if __name__ == '__main__':
    #train(HYPERPARAMETERS)
    embed()