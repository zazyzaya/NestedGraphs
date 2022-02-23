from copy import deepcopy
import os 
import pickle
import sys 
import time 

import torch 
import torch.distributed as dist
import torch.distributed.rpc as rpc 
import torch.multiprocessing as mp
from torch.nn import BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam 

from emb_train import EMBED_SIZE
from models.anom_gan import NodeGeneratorTopology, NodeGenerator,\
    GATDiscriminator, GCNDiscriminator, DualGCNDiscriminator,\
    DualGATDiscriminator

N_JOBS = 4 # How many worker processes will train the model
P_THREADS = 4 # About the point of diminishing returns from experiments

criterion = BCEWithLogitsLoss()
TRUE_VAL = 0.01
FALSE_VAL = 0.99

SKIP_GEN = 1
WARMUP = -1
PATIENCE = 100

G_HIDDEN_DIM = 128
NOISE_DIM = 128

LR=0.001
HIDDEN_DIM = 32
KWARGS = {
    'heads': 8
}

Disc = GATDiscriminator

def no_gen_step(embs, graph, disc):
    # Positive samples & train embedder
    t_preds = disc.forward(embs, graph)

    # Negative samples
    fake = embs[torch.randperm(embs.size(0))]
    f_preds = disc.forward(fake, graph)
    
    # Calculate loss (combine true and false so only need
    # one call to backward)
    preds = torch.cat([t_preds, f_preds], dim=1)
    labels = torch.cat([
        torch.full(t_preds.size(), TRUE_VAL),
        torch.full(f_preds.size(), FALSE_VAL)
    ], dim=1)

    return criterion(preds, labels)

def data_split(graphs, workers):
    min_jobs = [len(graphs) // workers] * workers
    remainder = len(graphs) % workers 
    for i in range(remainder):
        min_jobs[i] += 1

    jobs = [0]
    for j in min_jobs:
        jobs.append(jobs[-1] + j)

    return jobs

def proc_job(rank, world_size, all_graphs, jobs, val, epochs=250):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Sets number of threads used by this worker
    torch.set_num_threads(P_THREADS)

    my_graphs=[]; my_embs=[]
    for gid in range(jobs[rank], jobs[rank+1]):
        with open(DATA_HOME + 'graph%d.pkl' % all_graphs[gid], 'rb') as f:
            my_graphs.append(pickle.load(f))
        with open(DATA_HOME + 'emb%d.pkl' % all_graphs[gid], 'rb') as f:
            my_embs.append(pickle.load(f))

    with open(DATA_HOME + 'graph%d.pkl' % (val+rank), 'rb') as f:
        val_graph = pickle.load(f)
    with open(DATA_HOME + 'emb%d.pkl' % (val+rank), 'rb') as f:
        val_emb = pickle.load(f)

    disc = Disc(EMBED_SIZE, my_graphs[0].x.size(1), HIDDEN_DIM, **KWARGS)

    # Initialize shared models
    disc = DDP(disc)

    # Initialize optimizer
    d_opt = Adam(disc.parameters(), lr=LR)
    
    num_samples = len(my_graphs)

    best = float('inf')
    no_improvement = 0 

    for e in range(epochs):
        st = time.time()
        
        disc.train()
        #gen.train()

        # Train disc every time
        d_opt.zero_grad()
        d_loss_num = 0
        for i in range(num_samples):
            st = time.time() 
            d_loss = no_gen_step(my_embs[i], my_graphs[i], disc)
            d_loss.backward()
            d_loss_num += d_loss.item()

        d_opt.step() 
        d_loss_num /= num_samples

        # Validation step 
        with torch.no_grad():
            disc.eval()
            val_loss = no_gen_step(val_emb, val_graph, disc)

        # Synchronize value across all machines
        dist.all_reduce(val_loss)

        if rank == 0:
            print(
                '[%d] Disc Loss: %0.4f  Val Loss: %0.4f (%0.2f)' % 
                (e, d_loss_num, val_loss.item(), time.time()-st)
            )

        if val_loss < best or e < WARMUP:
            best = val_loss 
            no_improvement = 0

            if rank == 0:
                torch.save(disc.module, 'saved_models/detector/disc.pkl')

        else:
            no_improvement += 1
            if no_improvement > PATIENCE:
                if rank == 0:
                    print("Early stopping!")
                break            

        dist.barrier()

    dist.barrier()
    if rank == 0:
        dist.destroy_process_group()


TRAIN_GRAPHS = [i for i in range(1,13)]
DATA_HOME = 'inputs/benign/'
def main():
    world_size = min(N_JOBS, len(TRAIN_GRAPHS))
    jobs = data_split(TRAIN_GRAPHS, world_size)

    mp.spawn(proc_job,
        args=(world_size,TRAIN_GRAPHS,jobs,13),
        nprocs=world_size,
        join=True)

if __name__ == '__main__':
    main()