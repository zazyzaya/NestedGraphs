import os 
import time
from types import SimpleNamespace
import pickle 

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc 
import torch.multiprocessing as mp
from torch import nn 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam 
from tqdm import tqdm

from models.embedder import NodeEmbedderSelfAttention
from dist_train import data_split, TRAIN_GRAPHS

DAY = 23
WORKERS = 4
P_THREADS = 4
DATA_HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/Sept%d/benign/'%DAY

HYPERPARAMS = SimpleNamespace(
    emb_hidden=256, emb_t_hidden=1024, emb_out=256,
    attn_kw = {'heads': 8, 'layers': 4}, t2v=64,
    emb_lr=0.0001, epochs=10000, 
    max_samples=128, batch_size=64
)

def sample(nodes, batch_size=64, max_samples=128, rnd=True):
    if rnd:
        batches = torch.randperm(nodes.num_nodes)
    else:
        batches = torch.arange(nodes.num_nodes)
    
    batches = batches.unfold(dimension=0, size=batch_size, step=batch_size)
    
    i=0
    for b in batches:
        yield {
            'regs': nodes.sample_feat('regs', batch=b, max_samples=max_samples),
            'files': nodes.sample_feat('files', batch=b, max_samples=max_samples)
        }
        
def sample_one(nodes, batch_size=64, max_samples=256):
    '''
    Grab a random sample of nodes from the graph
    '''
    return next(sample(nodes, batch_size, max_samples))

def sample_all(nodes, max_samples=128):
    '''
    Samples every node with the full sequence in order
    '''
    return next(sample(nodes, batch_size=nodes.num_nodes, max_samples=max_samples, rnd=False))

def emb_step(emb, data, tau=0.05):
    '''
    Uses self-contrastive learning with the dropout that's already there
    to produce embeddings that are as self-similar as possible

    https://arxiv.org/pdf/2104.08821.pdf

              sim(x_i, x'_i)
    L = ----------------------------
         Sum(j=/=i) sim(x_i, x'_j)
    '''

    # Pass the same data through the net twice to get embeddings with
    # different dropout masks 
    a = emb(data); b = emb(data)
    
    # Compute cosine sim manually 
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    res = torch.exp(res/tau)

    # Self similarity (should be high) is diagonal
    # Dissimilarity to other samples is what remains
    pos = res.diagonal()
    neg = (res.sum(dim=1)-pos)+1e-9

    return (-torch.log(pos/neg)).mean()


def train_loop(rank, all_graphs, jobs, hp, is_dist=True):
    my_graphs=[]; my_nodes=[]
    for gid in range(jobs[rank], jobs[rank+1]):
        with open(DATA_HOME + 'graph%d.pkl' % all_graphs[gid], 'rb') as f:
            my_graphs.append(pickle.load(f))
        with open(DATA_HOME + 'nodes%d.pkl' % all_graphs[gid], 'rb') as f:
            my_nodes.append(pickle.load(f))

    emb = NodeEmbedderSelfAttention(
        my_nodes[0].file_dim, 
        my_nodes[0].reg_dim, 
        hp.emb_hidden, hp.emb_t_hidden, hp.emb_out,
        attn_kw=hp.attn_kw,
        t2v_dim=hp.t2v
    )

    # For debugging without multiprocessing
    if is_dist:
        emb = DDP(emb)

    opt = Adam(emb.parameters(), lr=hp.emb_lr)
    emb.train()

    num_samples = len(my_graphs)

    # Best loss 
    best = float('inf')
    for e in range(hp.epochs):
        for i,j in enumerate(torch.randperm(num_samples)):
            st = time.time() 

            opt.zero_grad()
            data = sample_one(my_nodes[j], batch_size=hp.batch_size, max_samples=hp.max_samples)

            #prog = tqdm(total=my_nodes[j].num_nodes//hp.batch_size)
            #for data in data_loader:
            loss = emb_step(emb, data)
            loss.backward()
            opt.step()

            # Synchronize loss vals across workers
            if is_dist:
                dist.all_reduce(loss, op=dist.ReduceOp.MAX)

            if rank==0:
                print('[%d-%d] Loss: %0.4f    (%0.2fs)' % (e,i,loss.item(),time.time()-st))

        # Purely unsupervised, so just save model periodically, assuming 
        # it's getting better with time
        dist.barrier() 
        if rank==0: 
            torch.save(
                (emb.module.state_dict(), emb.module.args, emb.module.kwargs), 
                'saved_models/embedder/emb.pkl'
            )

        dist.barrier()
    

def proc_job(rank, world_size, all_graphs, jobs, hp):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Sets number of threads used by this worker
    torch.set_num_threads(P_THREADS)
    
    train_loop(rank, all_graphs, jobs, hp)

    # Wait for everyone to finish training
    dist.barrier()

    # Cleanup
    if rank == 0:
        dist.destroy_process_group()

if __name__ == '__main__':
    world_size = min(WORKERS, len(TRAIN_GRAPHS))
    jobs = data_split(TRAIN_GRAPHS, world_size)

    mp.spawn(proc_job,
        args=(world_size,TRAIN_GRAPHS,jobs,HYPERPARAMS),
        nprocs=world_size,
        join=True)

    #train_loop(0,TRAIN_GRAPHS,jobs,HYPERPARAMS,is_dist=False)