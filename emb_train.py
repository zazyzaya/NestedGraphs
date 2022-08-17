import glob
import sys 
import time
import pickle
from types import SimpleNamespace

import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from models.embedder import NodeEmbedderSelfAttention, NodeDecoder

FEAT_DIM = {
    'files': 37,
    'regs': 35
}

if len(sys.argv) > 1:
    FEAT = sys.argv[1]
else:
    FEAT = 'files'

torch.set_num_threads(8)

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'
HYPERPARAMS = SimpleNamespace(
    e_hidden=256, t_hidden=1024, out=256,
    heads=8, e_layers=4, t2v=64,
    d_hidden=1024, d_layers=3,
    e_lr=0.001, d_lr=0.01, epochs=1000, 
    max_samples=512, batch_size=256, mean=True
)

criterion = BCEWithLogitsLoss()
def sample(nodes, batch_size=64, max_samples=128, rnd=True):
    if rnd:
        batches = torch.randperm(nodes.num_nodes)
    else:
        batches = torch.arange(nodes.num_nodes)
    
    batches = batches.unfold(dimension=0, size=batch_size, step=batch_size)
    
    for b in batches:
        yield nodes.sample_feat(FEAT, batch=b, max_samples=max_samples)

def sample_one(nodes, batch_size=64, max_samples=128): 
    return next(sample(nodes, batch_size, max_samples))


def step(emb, decoder, ts, seq, samples=64):
    '''
    Generates random groups of features from the input sequence
    B.c. when you think about it, we're just trying to get it
    good at summarizing what's in the sequences, not what's in 
    sequences observed in processes. If it can do one, it can do the other(?)
    '''
    t_idx = torch.randperm(ts.data.size(0)).unfold(dimension=0, step=samples, size=samples)
    f_idx = torch.randperm(ts.data.size(0)).unfold(dimension=0, step=samples, size=samples)

    # Make sure L x B x d order
    # Kind of silly to repack it, but I can't think of a better 
    # way so this still works for embedding the real data 
    # down the pipeline
    t = pack_padded_sequence(ts.data[t_idx].transpose(0,1), [samples]*t_idx.size(0))
    x = pack_padded_sequence(seq.data[t_idx].transpose(0,1), [samples]*t_idx.size(0))
    cls = emb(t,x)
    t_scores = decoder(cls, t, x)

    f_t = pack_padded_sequence(ts.data[f_idx].transpose(0,1), [samples]*f_idx.size(0))
    f_x = pack_padded_sequence(seq.data[f_idx].transpose(0,1), [samples]*f_idx.size(0))
    f_scores = decoder(cls, f_t, f_x)

    labels = torch.cat(
        [torch.full(t_scores.size(),1), torch.zeros(f_scores.size())],
        dim=0
    )
    scores = torch.cat([t_scores, f_scores], dim=0)
    return criterion(scores, labels)

def train(hp):
    enc = NodeEmbedderSelfAttention(
        FEAT_DIM[FEAT], 
        hp.e_hidden, hp.t_hidden, hp.out,
        hp.heads, hp.e_layers,
        t2v_dim=hp.t2v, mean=hp.mean
    )

    dec = NodeDecoder(
        hp.out, FEAT_DIM[FEAT], 
        hp.d_hidden, hp.d_layers, 
        t2v_dim=hp.t2v
    )

    opts = [
        Adam(enc.parameters(), lr=hp.e_lr), 
        Adam(dec.parameters(), lr=hp.d_lr)
    ]

    nodes = []
    for fname in glob.glob(HOME+'inputs/**/benign/nodes*.pkl'):
        with open(fname, 'rb') as f:
            nodes.append(pickle.load(f))

    for e in range(hp.epochs):
        st = time.time()
        [o.zero_grad() for o in opts]

        ts,x = sample_one(nodes[e%len(nodes)], hp.batch_size, hp.max_samples)
        loss = step(enc, dec, ts, x, hp.batch_size)
        loss.backward() 
        [o.step() for o in opts]

        print(
            '[%d] Loss %0.4f  (%0.2fs)' %
            (e, loss.item(), time.time()-st)
        )

        torch.save((enc, enc.args, enc.kwargs), 'saved_models/embedder/%s_emb.pkl' % FEAT)

if __name__ == '__main__':
    train(HYPERPARAMS)