import os 
import pickle
import sys 
import time
from numpy import r_ 

import torch 
import torch.distributed as dist
import torch.distributed.rpc as rpc 
import torch.multiprocessing as mp
from torch.nn import BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from models.hostlevel import NodeEmbedderSelfAttention, NodeEmbedderMultiSelfAttention
from models.emb_gan import GATDiscriminatorTime, NodeGeneratorCorrected, GATDiscriminator
from models.utils import kld_gauss

N_JOBS = 4 # How many worker processes will train the model
P_THREADS = 4 # About the point of diminishing returns from experiments

criterion = BCEWithLogitsLoss()

# Embedder params
EMB_HIDDEN = 64
EMB_OUT = 32
EMBED_SIZE = 64
T2V = 8
ATTN_MECH = 'torch'
ATTN_KW = {
    'layers': 2,
    'heads': 8
}

# GAN Params
HIDDEN_GEN = 128
TRUE_VAL = 0.0 # One-sided label smoothing
FALSE_VAL = 0.9 # False should approach 1 as in an anomaly score

# Training params
EMB_LR = 0.0005
MAX_SAMPLES = 50
PATIENCE = float('inf')

# Decide which architecture to use here
NodeEmb = NodeEmbedderSelfAttention
NodeGen = NodeGeneratorCorrected
NodeDisc = GATDiscriminator

def sample(nodes, max_samples=0):
    return {
        'regs': nodes.sample_feat('regs', max_samples=max_samples),
        'files': nodes.sample_feat('files', max_samples=max_samples)
    }

def train_step(nodes, graph, emb, gen, disc, e_opt, g_opt, d_opt):
    data = sample(nodes, max_samples=MAX_SAMPLES)

    # Positive samples & train embedder
    d_opt.zero_grad()
    e_opt.zero_grad()
    
    # Improve stability
    emb.train()
    disc.train()
    gen.eval()

    embs = emb.forward(data)
    t_preds = disc.forward(embs, graph)

    # Negative samples
    fake = gen.forward(graph).detach()
    f_preds = disc.forward(fake, graph)

    # Random samples
    rand = torch.rand(graph.x.size(0), EMBED_SIZE)
    r_preds = disc.forward(rand, graph)
    
    # Calculate loss (combine true and false so only need
    # one call to backward)
    preds = torch.cat([t_preds, f_preds, r_preds], dim=1)
    labels = torch.cat([
        torch.full(t_preds.size(), TRUE_VAL),
        torch.full(f_preds.size(), FALSE_VAL),
        torch.full(r_preds.size(), FALSE_VAL)
    ], dim=1)

    d_loss = criterion(preds, labels)
    d_loss.backward()
    d_opt.step()
    e_opt.step()

    # Train generator
    g_opt.zero_grad()
    emb.eval()
    disc.eval()
    gen.train()

    fake, mu, std = gen(graph)
    
    preds = disc.forward(fake, graph)
    g_loss = criterion(preds, torch.full((graph.num_nodes,1), TRUE_VAL))
    g_loss += kld_gauss(mu,std)

    g_loss.backward()
    g_opt.step()
    print("Gen Step")

    return d_loss.item(), g_loss.item()


def data_split(graphs, workers):
    min_jobs = [len(graphs) // workers] * workers
    remainder = len(graphs) % workers 
    for i in range(remainder):
        min_jobs[i] += 1

    jobs = [0]
    for j in min_jobs:
        jobs.append(jobs[-1] + j)

    return jobs

def proc_job(rank, world_size, all_graphs, jobs, val, epochs=100):
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

    with open(DATA_HOME + 'graph%d.pkl' % (val+rank), 'rb') as f:
        val_graph = pickle.load(f)
    with open(DATA_HOME + 'nodes%d.pkl' % (val+rank), 'rb') as f:
        val_nodes = pickle.load(f)

    emb = NodeEmb(
        my_nodes[0].file_dim, 
        my_nodes[0].reg_dim,
        my_nodes[0].mod_dim, 
        EMB_HIDDEN, EMB_OUT,
        EMBED_SIZE,
        t2v_dim=T2V,
        attn=ATTN_MECH,
        attn_kw=ATTN_KW
    )
    gen = NodeGen(my_graphs[0].x.size(1), 32, HIDDEN_GEN, EMBED_SIZE)
    disc = NodeDisc(EMBED_SIZE, 16, heads=16)

    # Initialize shared models
    gen = DDP(gen)
    emb = DDP(emb)
    disc = DDP(disc)

    # Initialize optimizers
    e_opt = Adam(emb.parameters(), lr=EMB_LR)
    d_opt = Adam(disc.parameters(), lr=0.01)
    g_opt = Adam(gen.parameters(), lr=0.01)
    
    best = float('inf')
    num_samples = len(my_graphs)
    early_stop = False
    for e in range(epochs):
        # Break in validatation section only gets out 
        # of inner the loop; update this variable to 
        # fully break
        if early_stop:
            break 

        for i in range(num_samples):
            st = time.time() 
            
            emb.train()
            gen.train() 
            disc.train()

            d_loss,g_loss = train_step(
                my_nodes[i], my_graphs[i],
                emb, gen, disc,
                e_opt, g_opt, d_opt
            ) 
   
            if rank == 0:
                print(
                    "[%d-%d] Disc: %0.4f, Gen: %0.4f (%0.2fs)" 
                    % (e, i, d_loss, g_loss, time.time()-st)
                )

            # Validation step 
            with torch.no_grad():
                emb.eval()
                disc.eval()
                gen.eval()
                
                data = sample(val_nodes, max_samples=MAX_SAMPLES)
                
                embs = emb.forward(data)
                t_preds = disc.forward(embs, val_graph)
                
                fake = gen.forward(val_graph)
                f_preds = disc.forward(fake, val_graph)

                preds = torch.cat([t_preds, f_preds], dim=1)
                labels = torch.cat([
                    torch.full(t_preds.size(), TRUE_VAL),
                    torch.full(f_preds.size(), FALSE_VAL)
                ], dim=1)

                val_loss = criterion(preds, labels)

        
            # Synchronize value across all machines
            dist.all_reduce(val_loss)

            if rank == 0:
                print("Validation loss: %0.4f" % (val_loss.item()/world_size), end='')

            if val_loss < best:
                # Reset counter if pretty close to old best
                no_improvement = 0
                best = val_loss

                # But only save the model if it's actually better
                if rank == 0:
                    print("*")
                    torch.save(disc.module, 'saved_models/embedder/disc_0.pkl')
                    torch.save(gen.module, 'saved_models/embedder/gen_0.pkl')
                    torch.save(emb.module, 'saved_models/embedder/emb_0.pkl')

            else:
                no_improvement += 1
                if rank == 0:
                    print() 

                if no_improvement > PATIENCE:
                    if rank == 0:
                        print("Early stopping!")
                    
                    early_stop = True
                    break     

        # So we can check in and see if the validation is really worth it
        # Always save, just have the filename different so we can differ what
        # val thought was best, vs. what just training forever looks like
        if rank == 0:
            torch.save(disc.module, 'saved_models/embedder/disc_1.pkl')
            torch.save(gen.module, 'saved_models/embedder/gen_1.pkl')
            torch.save(emb.module, 'saved_models/embedder/emb_1.pkl')

        dist.barrier()

    dist.barrier()
    
    # Cleanup
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