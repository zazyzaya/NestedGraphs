import itertools
import os 
import pickle
import time
import json
from types import SimpleNamespace 

import torch 
import torch.distributed as dist
import torch.distributed.rpc as rpc 
import torch.multiprocessing as mp
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Adagrad

from models.hostlevel import NodeEmbedderSelfAttention, NodeEmbedderNoTime
from models.emb_gan import GATDiscriminator, NodeGeneratorCorrected, GCNDiscriminator, \
	NodeGeneratorNonVariational, TreeGRUDiscriminator, TreeGRUGenerator, FFNNDiscriminator
from models.utils import kld_gauss
from gan_test import test_emb_input

N_JOBS = 4 # How many worker processes will train the model
P_THREADS = 4 # About the point of diminishing returns from experiments

criterion = BCEWithLogitsLoss()
mse = MSELoss()

# Embedder params
EMB_HIDDEN = 64
EMB_OUT = 32
EMB_SIZE = 64
T2V = 8
ATTN_MECH = 'torch'
ATTN_KW = {
	'layers': 4,
	'heads': 8
}

# Gen Params
GEN_HIDDEN = 128
GEN_LATENT = 16
QUANTILE = 1.

# Disc params
DISC_HIDDEN = 8 # Optimal hyperparam search
TRUE_VAL = 0.1 # One-sided label smoothing (real values smoothed)
FALSE_VAL = 1. # False should approach 1 as in an anomaly score

# Training params
EPOCHS = 5
BOOSTED = 0
EMB_LR = 0.005
GEN_LR = 0.005
DISC_LR= 0.005
MAX_SAMPLES = 50
WD = 0.0

# Decide which architecture to use here
NodeEmb = NodeEmbedderSelfAttention
NodeGen = NodeGeneratorNonVariational
NodeDisc = GCNDiscriminator

def sample(nodes, max_samples=0):
	return{
		'regs': nodes.sample_feat('regs', max_samples=max_samples),
		'files': nodes.sample_feat('files', max_samples=max_samples)
	}

def get_emb(nodes, emb, ms=MAX_SAMPLES):
	data = sample(nodes, max_samples=ms)
	
	with torch.no_grad():
		emb.eval()
		zs = emb.forward(data)

	return zs 

def get_cutoff(graph, nodes, emb, disc):
	emb.eval()
	disc.eval()
	
	with torch.no_grad():
		zs = get_emb(nodes, emb)
		preds = disc.forward(zs, graph)

	return torch.quantile(preds, 0.999)

def emb_step(data, nodes, graph, emb, gen, disc, e_opt, g_opt, d_opt):
	# Positive samples & train embedder
	e_opt.zero_grad()
	
	# Improve stability
	emb.train()
	disc.eval()

	# Positive samples
	embs = emb.forward(data)
	t_preds = disc.forward(embs, graph)
	
	labels = torch.full(t_preds.size(), TRUE_VAL)
	loss = criterion(t_preds, labels)

	loss.backward()
	e_opt.step()

	return loss

def gen_step(z, data, nodes, graph, emb, gen, disc, e_opt, g_opt, d_opt, verbose=True):
	'''
	Implimenting OCAN "bad generator" loss function for anomaly detection
	https://arxiv.org/abs/1803.01798
	'''
	
	g_opt.zero_grad()
	
	emb.eval()
	disc.eval()
	gen.train()

	fake = gen(graph)
	g_loss = mse(fake, z)

	# Paper only runs BCE loss on predictions with 
	# low probability. I.e., if a sample looks "fake enough", 
	# right on the boarder of the discriminator's decision boundary
	# don't change anything other than the MSE loss 
	with torch.no_grad():
		real = disc(z, graph)
		cutoff = torch.quantile(real, QUANTILE)

	# Training as a "bad GAN". Results should look fake, but be as 
	# close to the real distribution as possible
	preds = disc(fake, graph)

	# Push the lowest out, pull the highest back in 
	tr_preds = preds[preds <= cutoff]
	fk_preds = preds[preds >  cutoff]

	if verbose:
		print("%0.2f%% Low/High ratio" % (100*tr_preds.size(0)/preds.size(0)))

	# If one worker runs backward on BCE loss, all of them must,
	# otherwise DDP throws a hissy fit
	#bce_loss = torch.tensor(tr_preds.size(0))
	#dist.all_reduce(bce_loss)

	#if bce_loss and tr_preds.size(0):
	labels = torch.cat([
		torch.full(tr_preds.size(), FALSE_VAL),
		torch.full(fk_preds.size(), TRUE_VAL)
	], dim=0)

	preds = torch.cat([
		tr_preds, fk_preds
	], dim=0)
	g_loss += criterion(preds, labels)
	
	'''
	# Not a great solution, but the other workers have to make
	# some updates using the disc gradient, otherwise DDP throws an 
	# error
	elif bce_loss:
		g_loss += criterion(
			preds[preds.argmin()], 
			torch.tensor([FALSE_VAL])
		)
	'''

	g_loss.backward()
	g_opt.step()
	
	return g_loss

def disc_step(z, data, nodes, graph, emb, gen, disc, e_opt, g_opt, d_opt):
	# Positive samples & train embedder
	d_opt.zero_grad()
	
	# Improve stability
	emb.eval()
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


def data_split(graphs, workers):
	min_jobs = [len(graphs) // workers] * workers
	remainder = len(graphs) % workers 
	for i in range(remainder):
		min_jobs[i] += 1

	jobs = [0]
	for j in min_jobs:
		jobs.append(jobs[-1] + j)

	return jobs

def proc_job(rank, world_size, all_graphs, jobs, val, hp):
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

	if rank == 0:
		print(hp)

		with open(TEST_HOME + 'graph201.pkl', 'rb') as f:
			test_graph = pickle.load(f)
		with open(TEST_HOME + 'nodes201.pkl', 'rb') as f:
			test_nodes = pickle.load(f)
		with open('testlog.txt', 'w+') as f:
			f.write('AUC\tAP\tE-loss\tD-loss\tG-loss\n')

	emb = NodeEmb(
		my_nodes[0].file_dim, 
		my_nodes[0].reg_dim,
		my_nodes[0].mod_dim, 
		hp.emb_hidden, hp.emb_out, hp.emb_size,
		attn=hp.attn_mech, attn_kw=hp.attn_kw,
		t2v_dim=hp.t2v
	)
	gen = NodeGen(
		my_graphs[0].x.size(1), 
		hp.gen_latent, 
		hp.gen_hidden, 
		hp.emb_size
	)

	disc = NodeDisc(hp.emb_size, hp.disc_hidden)

	# Initialize shared models
	gen = DDP(gen)
	emb = DDP(emb)
	disc = DDP(disc)

	# Initialize optimizers
	e_opt = Adam(emb.parameters(), lr=hp.emb_lr, betas=(0.5, 0.999), weight_decay=hp.wd)
	d_opt = Adam(disc.parameters(), lr=hp.disc_lr, betas=(0.5, 0.999), weight_decay=hp.wd)
	g_opt = Adam(gen.parameters(), lr=hp.gen_lr, betas=(0.5, 0.999), weight_decay=hp.wd)

	num_samples = len(my_graphs)

	# Best all time
	all_time = 0
	all_time_str = ''

	for e in range(hp.epochs):
		for i,j in enumerate(torch.randperm(num_samples)):
			st = time.time() 
			
			emb.train()
			gen.train() 
			disc.train()

			data = sample(my_nodes[j], max_samples=MAX_SAMPLES)

			args = (
				data, my_nodes[j], my_graphs[j],
				emb, gen, disc,
				e_opt, g_opt, d_opt
			)

			z = get_emb(my_nodes[j], emb)
			d_loss = disc_step(z, *args)
			g_loss = gen_step(z, *args)
			e_loss = emb_step(*args)
   
			# Synchronize loss vals across workers
			dist.all_reduce(e_loss)
			dist.all_reduce(d_loss)
			dist.all_reduce(g_loss)

			# Average across workers (doesn't really matter, just for readability)
			e_loss/=N_JOBS; d_loss/=N_JOBS; g_loss/=N_JOBS

			if rank == 0:
				print(
					"[%d-%d] Emb: %0.4f Disc: %0.4f, Gen: %0.4f (%0.2fs)" 
					% (e, i, e_loss.item(), d_loss.item(), g_loss.item(), time.time()-st)
				)	

			# It's notoriously hard to validate GANs. Trying to see if theres
			# a good stopping point by looking at all this
			if rank == 0:
				torch.save(disc.module, 'saved_models/embedder/disc_1.pkl')
				torch.save(gen.module, 'saved_models/embedder/gen_1.pkl')
				torch.save(emb.module, 'saved_models/embedder/emb_1.pkl')

				test_z = get_emb(test_nodes, emb)
				auc,ap,pr = test_emb_input(test_z, test_nodes, test_graph, disc)

				print('AUC: ', auc, 'AP: ', ap)
				print("Top-k (Pr/Re):\n%s" % json.dumps(pr, indent=2))
				print()
				
				with open('testlog.txt', 'a+') as f:
					f.write('%f\t%f\t%f\t%f\t%f\n' % (auc,ap,e_loss,d_loss,g_loss))

				if ap > all_time: 
					all_time = ap 
					all_time_str = '%d-%d\t%f\t%f\t%f\t%f' % (e,i,auc, ap, pr[200][0], pr[200][1])

			dist.barrier()

		if hp.boosted > 0:
			# Just train disc and gen. Use static embeddings. Has been
			# shown to boost score considerably (also goes really fast)
			if rank==0:
				print("Prepare for liftoff...")

			my_embs = []
			for node in my_nodes:
				my_embs.append(get_emb(node, emb))

			print("Worker %d finished embedding" % rank)

			for e in range(hp.boosted):
				for i,j in enumerate(torch.randperm(num_samples)):
					args = (
						None, None, my_graphs[j],
						emb, gen, disc,
						e_opt, g_opt, d_opt
					)

					d_loss = disc_step(my_embs[j], *args)
					g_loss = gen_step(my_embs[j], *args, verbose=False)

					if rank == 0:
						print(
							"[%d-%d] Disc: %0.4f, Gen: %0.4f (%0.2fs)" 
							% (e, i, d_loss.item(), g_loss.item(), time.time()-st)
						)	

						auc,ap,pr = test_emb_input(test_z, test_nodes, test_graph, disc)
						print('AUC: ', auc, 'AP: ', ap)
						with open('testlog.txt', 'a+') as f:
							f.write('%f\t%f\t%f\t%f\t%f\n' % (auc,ap,-1,d_loss,g_loss))
			if rank == 0:
				print() 	

	if rank == 0:
		# Last 
		test_z = get_emb(test_nodes, emb)
		auc,ap,pr = test_emb_input(test_z, test_nodes, test_graph, disc)

		with open('final.txt', 'a+') as f:
			f.write(
				'%f\t%f\t%f\t%f\t%s\n'
				% (
					auc, ap, pr[200][0], pr[200][1],  
					all_time_str
				)
			)


	dist.barrier()
	
	# Cleanup
	if rank == 0:
		dist.destroy_process_group()


TRAIN_GRAPHS = [i for i in range(1,5)]
DATA_HOME = 'inputs/benign/'
TEST_HOME = 'inputs/mal/'
def main(params):
	# Setting globals doesn't work in multiproc
	# need to pass all hyperparams explicitly
	hyperparams = SimpleNamespace(
		emb_lr=EMB_LR, disc_lr=DISC_LR, gen_lr=GEN_LR,
		epochs=EPOCHS, boosted=BOOSTED, wd=WD,
		emb_hidden=EMB_HIDDEN, emb_out=EMB_OUT, emb_size=EMB_SIZE,
		t2v=T2V, attn_mech=ATTN_MECH, attn_kw=ATTN_KW, 
		gen_hidden=GEN_HIDDEN, gen_latent=GEN_LATENT, quantile=QUANTILE,
		disc_hidden=DISC_HIDDEN,
	)

	# Expects list of tuples (name, value) for gridsearching
	for param in params:
		assert hasattr(hyperparams, param[0]), 'Recieved value %s not in namespace' % param[0]
		setattr(hyperparams, param[0], param[1])

	world_size = min(N_JOBS, len(TRAIN_GRAPHS))
	jobs = data_split(TRAIN_GRAPHS, world_size)

	mp.spawn(proc_job,
		args=(world_size,TRAIN_GRAPHS,jobs,21,hyperparams),
		nprocs=world_size,
		join=True)

if __name__ == '__main__':
	with open('final.txt', 'w+') as f:
		f.write('Last AUC\tLast AP\tLast Pr\tLast Re\tBest Epoch\tBest AUC\tBest AP\tBest Pr\tBest Re\n')

	# List of values here
	params = {
		'quantile': [0.5,0.75,0.8,0.9,0.99,1.],
		'gen_latent': [16,32,64,8,4,0]
	}

	# Produces a nice list of [ [(p1, val), ..., (pn, val)], [...] ] for grid searches
	labels, values = zip(*params.items())
	combos = [v for v in itertools.product(*values)]
	args = [[(labels[i], c[i]) for i in range(len(c))] for c in combos]

	for arg in args:
		print(arg)
		with open('final.txt', 'a') as f:
			[f.write('%s: %f\n' % (v[0], v[1])) for v in arg]

		[main(arg) for _ in range(5)]
	
