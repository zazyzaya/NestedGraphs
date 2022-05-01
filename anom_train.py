from copy import deepcopy
import os 
import pickle
import json
import time 

import torch 
import torch.distributed as dist
import torch.distributed.rpc as rpc 
import torch.multiprocessing as mp
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from emb_train import sample
from gan_test import test_emb_input
from models.utils import reset_all_weights
from models.emb_gan import GCNDiscriminator

EMBED_SIZE = 1

N_JOBS = 4 # How many worker processes will train the model
P_THREADS = 4 # About the point of diminishing returns from experiments

criterion = BCEWithLogitsLoss()
mse = MSELoss()

TRUE_VAL = 0.1
FALSE_VAL = 1.

# Just need it to keep going a few epochs
# Too long and it overfits
EPOCHS = 100

D_LR=0.001
G_LR=0.001

def disc_step(z, graph, gen, disc, g_opt, d_opt):
    # Positive samples & train embedder
	d_opt.zero_grad()
	
	# Improve stability
	disc.train()
	gen.eval()

	# Positive samples
	t_preds = disc.forward(z, graph)

	# Negative samples
	fake = gen.forward(graph).detach()
	f_preds = disc.forward(fake, graph)

	# Random samples
	rand = torch.rand(graph.x.size(0), z.size(1))
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

	return d_loss

def gen_step(z, graph, gen, disc, g_opt, d_opt):
    g_opt.zero_grad()
	
    disc.eval()
    gen.train()

    fake = gen(graph)
    g_loss = mse(fake, z)

    #preds = disc(fake, graph)
    #labels = torch.full(preds.size(), 0.)
    #g_loss = criterion(preds, labels)

    g_loss.backward()
    g_opt.step()
	
    return g_loss

def data_split(graphs, workers):
    min_jobs = [len(graphs) // workers] * workers
    remainder = len(graphs) % workers 
    for i in range(remainder):
        min_jobs[i] += 1

    jobs = [0]
    for j in min_jobs:
        jobs.append(jobs[-1] + j)

    return jobs

def proc_job(rank, world_size, all_graphs, jobs, val):
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

    emb = torch.load('saved_models/embedder/emb_%d.pkl' % EMBED_SIZE)
    emb.eval()

    if rank == 0:
        with open('inputs/mal/graph201.pkl', 'rb') as f:
            test_graph = pickle.load(f)
        with open('inputs/mal/nodes201.pkl', 'rb') as f:
            test_nodes = pickle.load(f)
        with torch.no_grad():
            test_embs = emb(sample(test_nodes, max_samples=50))
    
    my_embs = []
    for nodes in my_nodes:
        with torch.no_grad():
            my_embs.append(emb(sample(nodes, max_samples=50)))

    del my_nodes 
    del emb 

    print("Worker %d finished loading" % rank)

    gen = torch.load('saved_models/embedder/gen_%d.pkl' % EMBED_SIZE) 
    disc = torch.load('saved_models/embedder/disc_%d.pkl' % EMBED_SIZE) 

    #reset_all_weights(gen)
    #reset_all_weights(disc)

    # Initialize shared models
    gen = DDP(gen)
    disc = DDP(disc)

    # Initialize optimizers
    d_opt = Adam(disc.parameters(), lr=D_LR)
    g_opt = Adam(gen.parameters(), lr=G_LR)
    
    with open('testlog.txt', 'w+') as f:
        f.write("AUC\tAP\tPr\tRe\n")

    num_samples = len(my_graphs)
    for e in range(EPOCHS):
        for i,j in enumerate(torch.randperm(num_samples)):
            st = time.time()
            args = (
                my_embs[j], my_graphs[j],
                gen, disc, g_opt, d_opt
            )

            g_loss = gen_step(*args)
            d_loss = disc_step(*args)

            if rank == 0:
                print(
                    '[%d-%d] Disc Loss: %0.4f  Gen Loss: %0.4f (%0.2fs)' % 
                    (e, i, d_loss, g_loss, time.time()-st),
                )

                auc,ap,pr = test_emb_input(test_embs, test_nodes, test_graph, disc)
                
                print('AUC: ', auc, 'AP: ', ap)
                #print("Top-k (Pr/Re):\n%s" % json.dumps(pr, indent=2))
                #print()

                with open('testlog.txt', 'a+') as f:
                    f.write('%f\t%f\t%f\t%f\n' % (auc, ap, pr[200][0], pr[200][1]))


    dist.barrier()
    if rank == 0:
        dist.destroy_process_group()


TRAIN_GRAPHS = [i for i in range(1,25)]
DATA_HOME = 'inputs/benign/'
def main():
    world_size = min(N_JOBS, len(TRAIN_GRAPHS))
    jobs = data_split(TRAIN_GRAPHS, world_size)

    mp.spawn(proc_job,
        args=(world_size,TRAIN_GRAPHS,jobs,21),
        nprocs=world_size,
        join=True)

if __name__ == '__main__':
    main()