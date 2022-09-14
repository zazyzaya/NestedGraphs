import glob
import os 
import pickle
import random
import socket
from types import SimpleNamespace 

import torch 
import torch.distributed as dist
import torch.distributed.rpc as rpc 
import torch.multiprocessing as mp
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from tqdm import tqdm 

from models.tgat import TGAT 
#from utils.graph_utils import get_src

N_JOBS = 12 # How many worker processes will train the model
P_THREADS = 1 # How many threads each worker gets

'''
Uniform batching
-----------------
J,T,time
1,8,76.84  (note htop shows max 5XX% utilization)
4,2,74.84
8,1,64.93
10,1,62.70
12,1,58.81
14,1,49.72
16,1,50.07

Degree-weighted batching
----------------
J,T,time
8,1,65.83
12,1,52.40
16,1,46.36
'''

DAY = 23 

# Depending on which machine we're running on 
if socket.gethostname() == 'colonial0':
    HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/'

# Note, this is running over sshfs so it may be slower to load
# may be worth it to make a local copy? 
elif socket.gethostname() == 'orion.ece.seas.gwu.edu':
    HOME = '/home/isaiah/code/NestedGraphs/inputs/'

HOME = HOME + 'Sept%d/benign/' % DAY 

hp = HYPERPARAMS = SimpleNamespace(
    tsize=64, hidden=64, heads=16, 
    emb_size=128, layers=3, nsize=64,
    epochs=100, lr=0.005
)

def fair_scheduler(n_workers, costs):
    '''
    There's probably a better way to do this, but for now
    just using greedy method. Costs is a list of tuples
    s.t. each tuple is (d,id) where d is the degree of node
    and id refers to the node id in the graph
    '''
    jobs = [[] for _ in range(n_workers)]
    labor_scheduled = [0] * n_workers

    while(costs):
        to_schedule = costs.index(max(costs, key=lambda x : x[0]))
        give_to = labor_scheduled.index(min(labor_scheduled))

        cost, nid = costs.pop(to_schedule)
        labor_scheduled[give_to] += cost
        jobs[give_to].append(nid)

    return jobs
        


def self_sim_step(model, graph, batch, tau=0.05): 
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
    a = model(graph, batch=batch)
    b = model(graph, batch=batch)
    
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


def generate_step(model, graph, batch, holdout=0.75):
    '''
    Uses dynamic link prediction as loss
    '''
    
    max_ts = torch.quantile(graph.edge_ts, holdout)
    z = model(graph, batch=batch, end_t=max_ts)

    ei = torch.cat([get_src(graph.edge_index, graph.csr_ptr), graph.edge_index], dim=0).long()
    future_edges = ei[:,graph.edge_ts > max_ts]

    # Can only check embeddings of nodes that were in batch
    known_src = (future_edges[0] == batch.unsqueeze(-1)).sum(dim=0)
    known_dst = (future_edges[1] == batch.unsqueeze(-1)).sum(dim=0)
    known_edges = future_edges[:, known_src.logical_and(known_dst)]



def proc_job(rank, world_size, hp):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Sets number of threads used by this worker
    torch.set_num_threads(P_THREADS)
    graphs = glob.glob(HOME+'full_graph*')
    
    if rank==0:
        print("Loading graph")
    with open(graphs[0],'rb') as f:
        g = pickle.load(f)

    tgat = TGAT(
        g.x.size(1), g.edge_feat_dim+1, 
        hp.tsize, hp.hidden, hp.emb_size, 
        hp.layers, hp.heads,
        neighborhood_size=hp.nsize,
        device=rank
    )

    tgat = DDP(tgat, device_ids=[rank])
    opt = Adam(tgat.parameters(), lr=hp.lr)
    loss = torch.tensor([float('nan')])
    for e in range(hp.epochs):
        random.shuffle(graphs)
        if rank==0:
            prog = tqdm(
                total=len(graphs), 
                desc='[%d-%d] Loss: %0.4f' % (e,0,loss.item())
            )

        for i,g_file in enumerate(graphs):
            with open(g_file,'rb') as f:
                g = pickle.load(f).to(rank)

            # Get this processes batch of jobs. In this case, 
            # nids of nodes that represent processes (x_n = [1,0,0,...,0])
            procs = (g.x[:,0] == 1).nonzero().squeeze(-1)
            
            '''
            naive method
            n_procs = procs.size(0)
            procs_per_worker = math.ceil(n_procs / world_size)
            my_batch = procs_per_worker*rank 
            my_batch = torch.arange(my_batch, my_batch+procs_per_worker)
            '''
            
            costs = [
                (min(g.get_one_hop(p.item())[0].size(0), hp.nsize), p.item())
                for p in procs
            ]
            my_batch = torch.tensor(fair_scheduler(world_size, costs)[rank]).to(rank)
            opt.zero_grad()
            loss = step(tgat, g, my_batch)
            loss.backward()
            opt.step() 
            
            if rank==0:
                prog.desc = '[%d-%d] Loss: %0.4f' % (e,i+1,loss.item())
                prog.update()

                torch.save(
                    (
                        tgat.module.state_dict(), 
                        tgat.module.args, 
                        tgat.module.kwargs
                    ), 'saved_models/tgat.pkl'
                )

            # Try to save some memory 
            del g,my_batch

        if rank==0:
            prog.close() 

    # Clean up 
    dist.barrier()
    if rank == 0:
        dist.destroy_process_group() 


def main(hp):
    world_size = 4
    mp.spawn(proc_job,
        args=(world_size,hp),
        nprocs=world_size,
        join=True)

if __name__ == '__main__':
    main(HYPERPARAMS)