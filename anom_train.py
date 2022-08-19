import itertools
import os 
import pickle
import time
import json
from types import SimpleNamespace 

import torch 
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Adagrad
from sklearn.model_selection import KFold

from models.gan import GCNDiscriminator, NodeGeneratorTopology, NodeGenerator, NodeGeneratorPerturb
from gan_test import score_many
from graph_utils import compress_ei

N_JOBS = 8 # How many worker processes will train the model
P_THREADS = 2 # How many threads each worker gets

TRAIN_GRAPHS = [i for i in range(1,25)]

# TODO incorporate all days
DAY = 23
DATA_HOME = 'inputs/Sept%d/benign/' % DAY
TEST_HOME = 'inputs/Sept%d/mal/' % DAY

criterion = BCEWithLogitsLoss()

ALPHA = 0.5
BETA = 1e-5
GAMMA = 0.1
DELTA = 0

HYPERPARAMS = SimpleNamespace(
    g_latent=64, g_hidden=256,
    d_hidden=64, 
    g_lr=0.00025, d_lr=0.00025, 
    epochs=25,
    alpha=ALPHA, beta=BETA, 
    gamma=GAMMA, delta=DELTA,
    emb_input=True
)

# Decide which architecture to use here
NodeGen = NodeGeneratorPerturb
NodeDisc = GCNDiscriminator

def gen_step(hp, nodes, graph, gen, disc):
    fake = gen(graph, nodes)
    f_preds = disc(fake, graph)

    labels = torch.full(f_preds.size(), hp.alpha)
    encirclement_loss = criterion(f_preds, labels)
    
    g_loss = encirclement_loss

    if hp.beta > 0: 
        mu = torch.stack([gen(graph, nodes) for _ in range(10)]).mean(dim=0)
        dispersion_loss = (1 / ((fake-mu).pow(2)+1e-9).mean(dim=1)).mean()
        g_loss += hp.beta*dispersion_loss

    # Only makes sense if we're doing the perturb gan
    if hp.delta > 0 and hp.emb_input:
        agitation_loss = (fake-nodes).pow(2).mean()
        g_loss += hp.delta*agitation_loss 
    
    return g_loss


def disc_step(hp, nodes, graph, gen, disc):
	# Positive samples
	t_preds = disc.forward(nodes, graph)

	# Negative samples
	fake = gen.forward(graph, nodes).detach()
	f_preds = disc.forward(fake, graph)

	t_loss = criterion(t_preds, torch.zeros(t_preds.size()))
	f_loss = criterion(f_preds, torch.full(f_preds.size(), 1.))

    # Optimize to identify real samples over fake samples
	d_loss = t_loss + hp.gamma*f_loss
	return d_loss


def data_split(graphs, workers):
	min_jobs = [len(graphs) // workers] * workers
	remainder = len(graphs) % workers 
	for i in range(remainder):
		min_jobs[i] += 1

	jobs = [0]
	for j in min_jobs:
		jobs.append(jobs[-1] + j)

	return jobs


def proc_job(rank, world_size, all_graphs, jobs, hp, val):
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
        
        my_nodes.append(torch.load(DATA_HOME + 'emb%d.pkl' % all_graphs[gid]))

    [compress_ei(g) for g in my_graphs]

    if rank == 0:
        print(hp)

    if hp.emb_input: 
        gen = NodeGeneratorPerturb(
            my_nodes[0].size(1),  
            hp.g_latent, 
            hp.g_hidden, 
            my_nodes[0].size(1) 
        )
    else: 
        gen = NodeGeneratorTopology(
            my_graphs[0].x.size(1), 
            hp.g_latent, 
            hp.g_hidden, 
            my_nodes[0].size(1) 
        )

    disc = NodeDisc(my_nodes[0].size(1), hp.d_hidden)

    # Initialize shared models
    gen = DDP(gen)
    disc = DDP(disc)

    # Initialize optimizers
    d_opt = Adam(disc.parameters(), lr=hp.d_lr, betas=(0.5, 0.999))
    g_opt = Adam(gen.parameters(), lr=hp.g_lr, betas=(0.5, 0.999))

    num_samples = len(my_graphs)

    # Best validation score
    best = 0.

    for e in range(hp.epochs):
        for i,j in enumerate(torch.randperm(num_samples)):
            st = time.time() 
            
            # Train generator
            gen.train(); disc.eval(); disc.requires_grad=False
            g_opt.zero_grad()
            g_loss = gen_step(
                hp, 
                my_nodes[j] if hp.emb_input else my_graphs[j], 
                my_graphs[j], 
                gen, disc
            )
            g_loss.backward()

            # Train discriminator
            gen.eval(); disc.train(); disc.requires_grad=True
            d_opt.zero_grad()
            d_loss = disc_step(hp, my_nodes[j], my_graphs[j], gen, disc)
            d_loss.backward()

            g_opt.step()
            d_opt.step()

            # Synchronize loss vals across workers
            dist.all_reduce(d_loss)
            dist.all_reduce(g_loss)

            # Average across workers (doesn't really matter, just for readability)
            d_loss/=N_JOBS; g_loss/=N_JOBS
            if rank == 0:
                out_str = ( 
                    "[%d-%d] Disc: %0.4f, Gen: %0.4f (%0.2fs)\n" 
                    % (e, i, d_loss.item(), g_loss.item(), time.time()-st)
                )

                auc, ap, pr_re = score_many(val, disc, DAY)
                out_str += 'AUC: %f  AP:  %f\n' % (auc,ap)
                out_str += json.dumps(pr_re, indent=1)

                print(out_str)

                with open('testlog.txt', 'a') as f:
                    f.write('%f\t%f\t%f\t%f\n' % (d_loss.item(), g_loss.item(), auc, ap))

                if ap > best: 
                    torch.save(
                        (disc.module.state_dict(), disc.module.args, disc.module.kwargs), 
                        'saved_models/detector/disc.pkl'
                    )
                    best = ap
            
            dist.barrier() 
    dist.barrier()

    # Cleanup
    if rank == 0:
        dist.destroy_process_group()


def kfold_validate(hp):
    world_size = min(N_JOBS, len(TRAIN_GRAPHS))
    jobs = data_split(TRAIN_GRAPHS, world_size)

    with open('results/out.txt', 'a+') as f:
        f.write(str(hp)+'\n\n')
        f.write('AUC\tAP\tTop-k Pr/Re\n')

    kfold = KFold(n_splits=5)
    test_graphs = torch.tensor([[201,402,660,104,205,321,255,355,503,462,559,419,609,771,955,874]]).T

    for te,va in kfold.split(test_graphs): 
        test = test_graphs[te].squeeze(-1)
        val = test_graphs[va].squeeze(-1)

        print("Testing:   ", test)
        print("Validating:", val)

        mp.spawn(proc_job,
            args=(world_size,TRAIN_GRAPHS,jobs,hp,val),
            nprocs=world_size,
            join=True)

        sd, args, kwargs = torch.load('saved_models/detector/disc.pkl')
        best_d = NodeDisc(*args, **kwargs)
        best_d.load_state_dict(sd)

        auc,ap,pr = score_many(test, best_d, DAY)
        print("AUC: ",auc,'AP: ',ap)
        print('Pr/Re')
        print(json.dumps(pr))

        with open('results/out.txt', 'a') as f:
            f.write('%f\t%f\t%s\n'%(auc,ap,json.dumps(pr)))

    with open('results/out.txt', 'a') as f:
        f.write('\n\n')
        
        

if __name__ == '__main__':
    '''
    world_size = min(N_JOBS, len(TRAIN_GRAPHS))
    jobs = data_split(TRAIN_GRAPHS, world_size)

    mp.spawn(proc_job,
        args=(world_size,TRAIN_GRAPHS,jobs,HYPERPARAMS,[201]),
        nprocs=world_size,
        join=True)
    '''
    hp = HYPERPARAMS
    for alpha in [0.9,0.7,0.5,0.3,0.1]:
        for beta in [0,1e-5,1e-3]:
            for gamma in [0.1,1]:
                hp.alpha = alpha 
                hp.gamma = gamma
                hp.beta = beta 

                kfold_validate(hp)