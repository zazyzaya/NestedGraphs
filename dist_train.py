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

from models.embedder import NodeEmbedderSelfAttention
from models.gan import GCNDiscriminator, NodeGenerator, NodeGeneratorTopology, NodeGeneratorPerturb
from models.utils import kld_gauss
from gan_test import test_emb_input

'''
Runtimes for various configurations:
Jobs,Threads
12,1 -> 128s/e
8,2 -> 110s/e
6,3 -> 117s/e
4,4 -> 130s/e
2,8 -> 146s/e
'''
N_JOBS = 8 # How many worker processes will train the model
P_THREADS = 2 # How many threads each worker gets

criterion = BCEWithLogitsLoss()
mse = MSELoss()

# Embedder params
EMB_HIDDEN = 64
EMB_T_HIDDEN = 256
EMB_OUT = 32
EMB_SIZE = 64
T2V = 64
ATTN_MECH = 'torch'
ATTN_KW = {
	'layers': 2,
	'heads': 8
}

# Gen Params
GEN_HIDDEN = 64
GEN_LATENT = 16
ALPHA = 0.5
BETA = 15

# Disc params
DISC_HIDDEN = 32 # Optimal hyperparam search
GAMMA = 0.1

# Training params
EPOCHS = 10
BOOSTED = 10
EMB_LR = 0.01
GEN_LR = 0.001
DISC_LR= 0.001
MAX_SAMPLES = 64
WD = 0.0
D_STEPS = 3

# Decide which architecture to use here
NodeEmb = NodeEmbedderSelfAttention
NodeGen = NodeGeneratorPerturb  # hyperparam
#GEN_TOPOLOGY = False
NodeDisc = GCNDiscriminator

def sample(nodes, batch_size=64, max_samples=128, rnd=True):    
	return {
		'regs': nodes.sample_feat('regs', batch=b, max_samples=max_samples),
        'files': nodes.sample_feat('files', batch=b, max_samples=max_samples)
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
	
	labels = torch.full(t_preds.size(), 0.0)
	loss = criterion(t_preds, labels)

	loss.backward()
	e_opt.step()

	return loss


def gen_step(z, data, nodes, graph, emb, gen, disc, e_opt, g_opt, d_opt, verbose=True, badgan=True):
	g_opt.zero_grad()
	
	emb.eval()
	disc.eval()
	gen.train()

	fake = gen(graph, z)
	f_preds = disc(fake, graph)

	labels = torch.full(f_preds.size(), ALPHA)
	encirclement_loss = criterion(f_preds, labels)
	
	#mu = fake.mean(dim=0)
	#dispersion_loss = (1 / ((fake-mu).pow(2)+1e-9)).mean()

	#print(encirclement_loss, dispersion_loss)
	agitation_loss = (fake-nodes).pow(2).mean()

	g_loss = encirclement_loss + agitation_loss #+ BETA*dispersion_loss
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
	fake = gen.forward(graph, z).detach()
	f_preds = disc.forward(fake, graph)

	t_loss = criterion(t_preds, torch.zeros(t_preds.size()))
	f_loss = criterion(f_preds, torch.full(f_preds.size(), 1.))
	
	# Added as an additional term in OCGAN paper
	#e_loss = -((1-t_preds)*torch.log(1-t_preds)).mean()

	d_loss = t_loss + GAMMA*f_loss
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

	NodeGen = NodeGeneratorTopology if hp.topo else NodeGenerator

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
		hp.emb_hidden, hp.emb_t_hidden, hp.emb_out, hp.emb_size,
		attn_kw=hp.attn_kw,
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

	# Best loss 
	best = float('inf')

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

			e_loss = emb_step(*args)
			z = get_emb(my_nodes[j], emb)
			
			g_loss = gen_step(z, *args, verbose=False)
			d_loss = [disc_step(z, *args) for _ in range(hp.d_steps)]
			d_loss = sum(d_loss)/hp.d_steps
   
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

					# Use 201 as validation set?
					torch.save(disc.module, 'saved_models/embedder/disc_0.pkl')
					torch.save(gen.module, 'saved_models/embedder/gen_0.pkl')
					torch.save(emb.module, 'saved_models/embedder/emb_0.pkl')

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
		
		# And save it for later, just in case
		torch.save(disc.module, 'saved_models/embedder/disc_1.pkl')
		torch.save(gen.module, 'saved_models/embedder/gen_1.pkl')
		torch.save(emb.module, 'saved_models/embedder/emb_1.pkl')

		disc = torch.load('saved_models/embedder/disc_0.pkl')
		emb = torch.load('saved_models/embedder/emb_0.pkl')

		test_z = get_emb(test_nodes, emb)
		s_auc, s_ap, s_pr = test_emb_input(test_z, test_nodes, test_graph, disc)

		with open('final.txt', 'a+') as f:
			f.write(
				'%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\n'
				% (
					auc, ap, pr[200][0], pr[200][1],
					s_auc, s_ap, s_pr[200][0], s_pr[200][1],    
					all_time_str
				)
			)


	dist.barrier()
	
	# Cleanup
	if rank == 0:
		dist.destroy_process_group()


TRAIN_GRAPHS = [i for i in range(1,25)]
DATA_HOME = 'inputs/benign/'
TEST_HOME = 'inputs/mal/'
def main(params):
	# Setting globals doesn't work in multiproc
	# need to pass all hyperparams explicitly
	hyperparams = SimpleNamespace(
		emb_lr=EMB_LR, disc_lr=DISC_LR, gen_lr=GEN_LR,
		epochs=EPOCHS, boosted=BOOSTED, wd=WD, d_steps=D_STEPS,
		emb_hidden=EMB_HIDDEN, emb_t_hidden=EMB_T_HIDDEN, emb_out=EMB_OUT, emb_size=EMB_SIZE,
		t2v=T2V, attn_mech=ATTN_MECH, attn_kw=ATTN_KW, 
		gen_hidden=GEN_HIDDEN, gen_latent=GEN_LATENT, topo=GEN_TOPOLOGY,
		disc_hidden=DISC_HIDDEN,
	)

	# Expects list of tuples (name, value) for gridsearching
	special = ['lr', 'attn.layers', 'attn.heads']
	for param in params:
		assert hasattr(hyperparams, param[0]) or param[0] in special, 'Recieved value %s not in namespace' % param[0]
		
		# Special case when all lr's are the same
		if param[0] == 'lr': 
			for lr in ['emb_lr', 'disc_lr', 'gen_lr']:
				setattr(hyperparams, lr, param[1])
		
		elif param[0] == 'attn.layers':
			kw  = hyperparams.attn_kw
			kw['layers'] = param[1]
			hyperparams.attn_kw = kw 
		
		elif param[0] == 'attn.heads':
			kw  = hyperparams.attn_kw
			kw['heads'] = param[1]
			hyperparams.attn_kw = kw

		else:
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
		'gen_latent': [8,16,32,64],
	}

	# Produces a nice list of [ [(p1, val), ..., (pn, val)], [...] ] for grid searches
	if params:
		labels, values = zip(*params.items())
		combos = [v for v in itertools.product(*values)]
		args = [[(labels[i], c[i]) for i in range(len(c))] for c in combos]
	else:
		args = [[]]

	for arg in args:
		print(arg)

		if arg:
			with open('final.txt', 'a') as f:
				[f.write('%s: %f\n' % (v[0], v[1])) for v in arg]

		[main(arg) for _ in range(5)]
	
