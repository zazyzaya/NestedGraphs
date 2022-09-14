import glob
import os 
import pickle
import random
import socket
from types import SimpleNamespace 

import torch 
import torch.distributed as dist
import torch.distributed.rpc as rpc 
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from tqdm import tqdm

from models.tgat import TGAT 
from models.leader_follower import Coordinator, WorkerDDP, get_worker
from utils.graph_utils import get_edge_index

DAY = 23 
P_THREADS = 1

DDP_PORT = '42069'
RPC_PORT = '22204'

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

def init_workers(num_workers, *args, **kwargs):
    rrefs = []
    for i in range(num_workers):
        rrefs.append(
            rpc.remote(
                i, get_worker, 
                args=args, 
                kwargs=dict(kwargs, **{'device': i})
            )
        )

    return rrefs 

def init_procs(rank, world_size, hp):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = DDP_PORT
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # RPC info
    options = rpc.TensorPipeRpcBackendOptions(
        init_method='tcp://localhost:' + RPC_PORT
    )

    # Start the server
    rpc.init_rpc(
        'worker'+str(rank), 
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )

    # Sets number of threads used by this worker
    torch.set_num_threads(P_THREADS)

    # Only need one coordinator
    if rank == 0:
        # Build a model for everyone (including yourself)
        graphs = glob.glob(HOME+'full_graph*')
        with open(graphs[0],'rb') as f:
            g = pickle.load(f)

        args = (
            g.x.size(1), g.edge_feat_dim+1, 
            hp.tsize, hp.hidden, hp.emb_size, 
            hp.layers, hp.heads,
        )
        kwargs = dict(neighborhood_size=hp.nsize)
        rrefs = init_workers(world_size, *args, **kwargs)
    
        # Also build the coordinator model which proc0 controls
        coordinator = Coordinator(
            rrefs, hp.nsize
        )

        # Then start training
        train(hp, coordinator, graphs)

    # Otherwise, just wait until everyone is done working
    rpc.shutdown()

def train(hp, model, graphs):
    opt = DistributedOptimizer(
        Adam, model.parameter_rrefs(), lr=hp.lr
    )
    loss = torch.tensor([float('nan')])
    for e in range(hp.epochs):
        random.shuffle(graphs)
        prog = tqdm(
            total=len(graphs), 
            desc='[%d-%d] Loss: %0.4f' % (e,0,loss.item())
        )

        for i,g_file in enumerate(graphs):
            with open(g_file,'rb') as f:
                g = pickle.load(f)

            with dist_autograd.context() as context_id:
                max_t = torch.quantile(g.edge_ts, 0.75)
                zs = model(g, end_t=max_t)
                loss = loss_fn(g,zs,max_t)
                loss.backward()
                opt.step() 
                
                prog.desc = '[%d-%d] Loss: %0.4f' % (e,i+1,loss.item())
                prog.update()
    
        prog.close()  

def dot(a,b):
    return (a*b).sum(dim=-1)

criterion = BCEWithLogitsLoss()
def loss_fn(g,zs,max_t):
    ei = get_edge_index(g)

    pos = ei[:,g.edge_ts >= max_t]
    neg = torch.randint(pos.min(), pos.max(), pos.size())

    pos = dot(zs[pos[0]],zs[pos[1]])
    neg = dot(zs[neg[0]],zs[neg[1]])

    p_y = torch.full(pos.size(), 1.)
    n_y = torch.zeros(neg.size())

    return criterion(
        torch.cat([pos,neg],dim=0),
        torch.cat([p_y,n_y],dim=0)
    )


def main(hp):
    world_size = 4
    mp.spawn(
        init_procs, 
        args=(world_size, hp),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main(HYPERPARAMS)