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
from train_single import cl_step, self_cl_step, mean_shifted_cl
#from utils.graph_utils import get_src

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

P_THREADS = 1
DAY = 23 
DEVICES = [2,3]

# Depending on which machine we're running on 
if socket.gethostname() == 'colonial0':
    HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/'

# Note, this is running over sshfs so it may be slower to load
# may be worth it to make a local copy? 
elif socket.gethostname() == 'orion.ece.seas.gwu.edu':
    HOME = '/home/isaiah/code/NestedGraphs/inputs/'

HOME = HOME + 'Sept%d/benign/' % DAY 

hp = HYPERPARAMS = SimpleNamespace(
    tsize=64, hidden=32, heads=8, 
    emb_size=128, layers=3, nsize=128,
    epochs=100, lr=0.0001
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
        g.x.size(1), g.edge_feat_dim, 
        hp.tsize, hp.hidden, hp.emb_size, 
        hp.layers, hp.heads,
        neighborhood_size=hp.nsize,
        device=DEVICES[rank]
    )

    tgat = DDP(tgat, device_ids=[DEVICES[rank]])
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
                g = pickle.load(f).to(DEVICES[rank])

            # Get this processes batch of jobs. In this case, 
            # nids of nodes that represent processes (x_n = [1,0,0,...,0])
            procs = (g.x[:,0] == 1).nonzero().squeeze(-1)
            
            costs = [
                (min(g.get_one_hop(p.item())[0].size(0), hp.nsize), p)
                for p in procs
            ]
            my_batch = torch.tensor(fair_scheduler(world_size, costs)[rank]).to(rank).long()
            opt.zero_grad()
            loss = mean_shifted_cl(tgat, g, my_batch)
            loss.backward()
            opt.step() 
            
            if rank==0:
                prog.desc = '[%d-%d] Loss: %0.4f' % (e,i+1,loss.item())
                prog.update()

                with open('log.txt', 'a') as f:
                    f.write('%f\n' % loss.item())

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
    with open('log.txt', 'w+'):
        pass 

    mp.spawn(proc_job,
        args=(len(DEVICES),hp),
        nprocs=len(DEVICES),
        join=True)

if __name__ == '__main__':
    main(HYPERPARAMS)