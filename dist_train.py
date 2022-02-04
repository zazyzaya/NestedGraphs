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

from models.hostlevel import NodeEmbedder
from models.gan import NodeGenerator, GATDescriminator

N_JOBS = 4 # How many worker processes will train the model
P_THREADS = 4 # About the point of diminishing returns from experiments

criterion = BCEWithLogitsLoss()
EMBED_SIZE = 16
TRUE_VAL = 0.
FALSE_VAL = 1. # False should be 1 as in an anomaly score


def train_step(nodes, graph, emb, gen, desc, e_opt, g_opt, d_opt):
    start = time.time() 
    data = nodes.sample()

    # Positive samples & train embedder
    e_opt.zero_grad()
    d_opt.zero_grad()
    real = emb.forward(data)
    preds = desc.forward(real, graph.x, graph.edge_index)
    r_loss = criterion(preds, torch.full((graph.num_nodes,1), TRUE_VAL))
    r_loss.backward()
    e_opt.step() 

    # Negative samples
    g_opt.zero_grad()
    fake = gen.forward(graph.x).detach()
    preds = desc.forward(fake, graph.x, graph.edge_index)
    f_loss = criterion(preds, torch.full((graph.num_nodes,1), FALSE_VAL))
    f_loss.backward() 
    d_opt.step() 

    # Train generator
    fake = gen(graph.x)
    preds = desc.forward(fake, graph.x, graph.edge_index)
    g_loss = criterion(preds, torch.full((graph.num_nodes,1), TRUE_VAL))
    g_loss.backward() 
    g_opt.step()

    elapsed = time.time() - start
    return r_loss.item(), f_loss.item(), g_loss.item(), elapsed


def data_split(graphs, workers):
    min_jobs = [len(graphs) // workers] * workers
    remainder = len(graphs) % workers 
    for i in range(remainder):
        min_jobs[i] += 1

    jobs = [0]
    for j in min_jobs:
        jobs.append(jobs[-1] + j)

    return jobs

def proc_job(rank, world_size, all_graphs, jobs, epochs=50):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.set_num_threads(P_THREADS)

    my_graphs=[]; my_nodes=[]
    for gid in range(jobs[rank], jobs[rank+1]):
        with open(DATA_HOME + 'graph%d.pkl' % all_graphs[gid], 'rb') as f:
            my_graphs.append(pickle.load(f))
        with open(DATA_HOME + 'nodes%d.pkl' % all_graphs[gid], 'rb') as f:
            my_nodes.append(pickle.load(f))

    emb = NodeEmbedder(
        my_nodes[0].file_dim, 
        my_nodes[0].reg_dim,
        my_nodes[0].mod_dim, 
        16, 8, EMBED_SIZE
    )
    gen = NodeGenerator(my_graphs[0].x.size(1), 32, 64, EMBED_SIZE)
    desc = GATDescriminator(EMBED_SIZE, my_graphs[0].x.size(1), 16)

    # Initialize shared models
    emb = DDP(emb)
    gen = DDP(gen)
    desc = DDP(desc)

    # Initialize optimizers
    e_opt = Adam(emb.parameters(), lr=0.01)
    g_opt = Adam(gen.parameters(), lr=0.01)
    d_opt = Adam(desc.parameters(), lr=0.01)

    num_samples = len(my_graphs)
    for e in range(epochs):
        for i in range(num_samples):
            r,f,g,elapsed = train_step(
                my_nodes[i], my_graphs[i],
                emb, gen, desc, 
                e_opt, g_opt, d_opt
            ) 

            if rank == 0:
                print(
                    "[%d] Emb: %0.4f, Gen: %0.4f, Disc: %0.4f (%0.2fs)" 
                    % (e, r, (r+f)/2, g, elapsed)
                )

        # Unfortunately, unbalanced data means some workers have to 
        # wait around for workers w more data to finish before continuing
        dist.barrier()

    if rank == 0:
        torch.save(emb.state_dict(), 'saved_models/emb.pkl')
        torch.save(desc.state_dict(), 'saved_models/desc.pkl')
        torch.save(gen.state_dict(), 'saved_models/gen.pkl')

        dist.destroy_process_group()


TRAIN_GRAPHS = [i for i in range(1,11)]
DATA_HOME = 'inputs/benign/'
def main():
    world_size = min(N_JOBS, len(TRAIN_GRAPHS))
    jobs = data_split(TRAIN_GRAPHS, world_size)

    mp.spawn(proc_job,
        args=(world_size,TRAIN_GRAPHS,jobs),
        nprocs=world_size,
        join=True)

if __name__ == '__main__':
    main()