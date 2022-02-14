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

from models.hostlevel import NodeEmbedderRNN, NodeEmbedderSelfAttention, NodeEmbedderAggr
from models.gan import NodeGenerator, GATDescriminator

N_JOBS = 4 # How many worker processes will train the model
P_THREADS = 4 # About the point of diminishing returns from experiments

criterion = BCEWithLogitsLoss()
EMBED_SIZE = 32
TRUE_VAL = 0.1 # Discourage every negative sample being -9999999
FALSE_VAL = 0.9 # False should approach 1 as in an anomaly score

# Decide which embedder to use here
NodeEmbedder = NodeEmbedderAggr

def train_step(nodes, graph, emb, gen, desc):
    data = nodes.sample()

    # Positive samples & train embedder
    real = emb.forward(data)
    preds = desc.forward(real, graph.x, graph.edge_index)
    r_loss = criterion(preds, torch.full((graph.num_nodes,1), TRUE_VAL))

    # Negative samples
    fake = gen.forward(graph.x).detach()
    preds = desc.forward(fake, graph.x, graph.edge_index)
    f_loss = criterion(preds, torch.full((graph.num_nodes,1), FALSE_VAL))

    # Train generator
    fake = gen(graph.x)
    preds = desc.forward(fake, graph.x, graph.edge_index)
    g_loss = criterion(preds, torch.full((graph.num_nodes,1), TRUE_VAL))
    
    return r_loss, f_loss, g_loss


def data_split(graphs, workers):
    min_jobs = [len(graphs) // workers] * workers
    remainder = len(graphs) % workers 
    for i in range(remainder):
        min_jobs[i] += 1

    jobs = [0]
    for j in min_jobs:
        jobs.append(jobs[-1] + j)

    return jobs

def proc_job(rank, world_size, all_graphs, jobs, epochs=250):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Sets number of threads used by this worker
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
    opts = [
        Adam(emb.parameters(), lr=0.01),
        Adam(desc.parameters(), lr=0.01),
        Adam(gen.parameters(), lr=0.01)
    ]

    num_samples = len(my_graphs)
    for e in range(epochs):
        for i in range(num_samples):
            st = time.time() 

            [o.zero_grad() for o in opts]

            r_loss,f_loss,g_loss = train_step(
                my_nodes[i], my_graphs[i],
                emb, gen, desc
            ) 

            # Only step after all data is processed
            # this acts as a barrier to prevent workers with 
            # fewer graphs from processing them more often    
            if rank == 0:
                print(
                    "[%d-%d] Emb: %0.4f, Disc: %0.4f, Gen: %0.4f (%0.2fs)" 
                    % (e, i, r_loss.item(), (r_loss+f_loss).item()/2, g_loss.item(), time.time()-st)
                )
                print("backward pass")
                st = time.time() 

            g_loss.backward() 
            f_loss.backward() 
            r_loss.backward()

            [o.step() for o in opts]
            print("%0.2fs" % (time.time() - st))

        if rank == 0:
            torch.save(emb.module, 'saved_models/emb.pkl')
            torch.save(desc.module, 'saved_models/desc.pkl')
            torch.save(gen.module, 'saved_models/gen.pkl')
        
        dist.barrier()

    dist.barrier()
    if rank == 0:
        torch.save(emb.module, 'saved_models/emb.pkl')
        torch.save(desc.module, 'saved_models/desc.pkl')
        torch.save(gen.module, 'saved_models/gen.pkl')

        dist.destroy_process_group()


TRAIN_GRAPHS = [i for i in range(1,13)]
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