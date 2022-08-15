import itertools
import os 
import pickle
import time
import json
from types import SimpleNamespace 

import torch 
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import Adam, Adagrad

from models.hostlevel import NodeEmbedderSelfAttention, NodeEmbedderNoTime
from models.emb_gan import GCNDiscriminator, NodeGenerator, NodeGeneratorTopology
from models.utils import kld_gauss
from gan_test import test_emb_input


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
QUANTILE = 0.8

# Disc params
DISC_HIDDEN = 8 # Optimal hyperparam search
TRUE_VAL = 0.1 # One-sided label smoothing (real values smoothed)
FALSE_VAL = 1. # False should approach 1 as in an anomaly score

# Training params
EPOCHS = 10
BOOSTED = 0
EMB_LR = 0.005
GEN_LR = 0.005
DISC_LR= 0.005
MAX_SAMPLES = 50
WD = 0.0

# Decide which architecture to use here
NodeEmb = NodeEmbedderSelfAttention
NodeGen = NodeGenerator
NodeDisc = GCNDiscriminator

torch.set_num_threads(16)

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

def gen_step(z, data, nodes, graph, emb, gen, disc, e_opt, g_opt, d_opt, verbose=False):
	'''
	Implimenting OCAN "bad generator" loss function for anomaly detection
	https://arxiv.org/abs/1803.01798
	'''
	# Don't want disc to train on real looking generated samples, 
	# so ensure gen's samples are near the decision boundary before
	# training loop ends
	bad = True
	while(bad):
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
		tr_preds = preds[preds <= cutoff]

		if verbose:
			print("%0.2f%% Low/High ratio" % (100*tr_preds.size(0)/preds.size(0)))

		if tr_preds.size(0):
			labels = torch.full(tr_preds.size(), FALSE_VAL)
			g_loss += criterion(tr_preds, labels)
		else:
			bad=False
		
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
	t_preds = torch.sigmoid(disc.forward(z, graph))

	# Negative samples
	fake = gen.forward(graph).detach()
	f_preds = torch.sigmoid(disc.forward(fake, graph))

	t_loss = -torch.log(1-t_preds).mean()
	f_loss = -torch.log(f_preds).mean()
	
	# Added as an additional term in OCGAN paper
	e_loss = -((1-t_preds)*torch.log(1-t_preds)).mean()

	d_loss = t_loss+f_loss+e_loss
	d_loss.backward()
	d_opt.step()

	return d_loss


def train_loop(all_graphs, val, hp):
	my_graphs=[]; my_nodes=[]
	for gid in all_graphs:
		with open(DATA_HOME + 'graph%d.pkl' % gid, 'rb') as f:
			my_graphs.append(pickle.load(f))
		with open(DATA_HOME + 'nodes%d.pkl' % gid, 'rb') as f:
			my_nodes.append(pickle.load(f))
	
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
			
			g_loss = gen_step(z, *args)
			d_loss = sum([disc_step(z, *args) for _ in range(hp.d_steps)])/hp.d_steps
			
			print(
				"[%d-%d] Emb: %0.4f Disc: %0.4f, Gen: %0.4f (%0.2fs)" 
				% (e, i, e_loss.item(), d_loss.item(), g_loss.item(), time.time()-st)
			)	

			# It's notoriously hard to validate GANs. Trying to see if theres
			# a good stopping point by looking at all this
			if g_loss.item() < best:
				best = g_loss.item()
				torch.save(disc, 'saved_models/embedder/disc_0.pkl')
				torch.save(gen, 'saved_models/embedder/gen_0.pkl')
				torch.save(emb, 'saved_models/embedder/emb_0.pkl')

			test_z = get_emb(test_nodes, emb)
			auc,ap,pr = test_emb_input(test_z, test_nodes, test_graph, disc)

			print('AUC: ', auc, 'AP: ', ap)
			#print("Top-k (Pr/Re):\n%s" % json.dumps(pr, indent=2))
			#print()
			
			with open('testlog.txt', 'a+') as f:
				f.write('%f\t%f\t%f\t%f\t%f\n' % (auc,ap,e_loss,d_loss,g_loss))

			if ap > all_time: 
				all_time = ap 
				all_time_str = '%d-%d\t%f\t%f\t%f\t%f' % (e,i,auc, ap, pr[200][0], pr[200][1])

		if hp.boosted > 0:
			# Just train disc and gen. Use static embeddings. Has been
			# shown to boost score considerably (also goes really fast)
			print("Prepare for liftoff...")

			my_embs = []
			for node in my_nodes:
				my_embs.append(get_emb(node, emb))

			print("Finished embedding")

			for e in range(hp.boosted):
				for i,j in enumerate(torch.randperm(num_samples)):
					args = (
						None, None, my_graphs[j],
						emb, gen, disc,
						e_opt, g_opt, d_opt
					)

					d_loss = disc_step(my_embs[j], *args)
					g_loss = gen_step(my_embs[j], *args, verbose=False)
					
					print(
						"[%d-%d] Disc: %0.4f, Gen: %0.4f (%0.2fs)" 
						% (e, i, d_loss.item(), g_loss.item(), time.time()-st)
					)	

					auc,ap,pr = test_emb_input(test_z, test_nodes, test_graph, disc)
					print('AUC: ', auc, 'AP: ', ap)
					with open('testlog.txt', 'a+') as f:
						f.write('%f\t%f\t%f\t%f\t%f\n' % (auc,ap,-1,d_loss,g_loss))
	
	# Last 
	test_z = get_emb(test_nodes, emb)
	auc,ap,pr = test_emb_input(test_z, test_nodes, test_graph, disc)
	
	# And save it for later, just in case
	torch.save(disc, 'saved_models/embedder/disc_1.pkl')
	torch.save(gen, 'saved_models/embedder/gen_1.pkl')
	torch.save(emb, 'saved_models/embedder/emb_1.pkl')

	with open('final.txt', 'a+') as f:
		f.write(
			'%f\t%f\t%f\t%f\t%s\n'
			% (
				auc, ap, pr[200][0], pr[200][1],    
				all_time_str
			)
		)



TRAIN_GRAPHS = [i for i in range(1,25)]
DATA_HOME = 'inputs/benign/'
TEST_HOME = 'inputs/mal/'
def main(params):
	# Setting globals doesn't work in multiproc
	# need to pass all hyperparams explicitly
	hyperparams = SimpleNamespace(
		emb_lr=EMB_LR, disc_lr=DISC_LR, gen_lr=GEN_LR,
		epochs=EPOCHS, boosted=BOOSTED, wd=WD, d_steps=1,
		emb_hidden=EMB_HIDDEN, emb_out=EMB_OUT, emb_size=EMB_SIZE,
		t2v=T2V, attn_mech=ATTN_MECH, attn_kw=ATTN_KW, 
		gen_hidden=GEN_HIDDEN, gen_latent=GEN_LATENT, quantile=QUANTILE,
		disc_hidden=DISC_HIDDEN,
	)

	# Expects list of tuples (name, value) for gridsearching
	for param in params:
		assert hasattr(hyperparams, param[0]) or param[0]=='lr', 'Recieved value %s not in namespace' % param[0]
		
		# Special case when all lr's are the same
		if param[0] == 'lr': 
			for lr in ['emb_lr', 'disc_lr', 'gen_lr']:
				setattr(hyperparams, lr, param[1])
		else:
			setattr(hyperparams, param[0], param[1])

	train_loop(TRAIN_GRAPHS,21,hyperparams)

if __name__ == '__main__':
	with open('final.txt', 'w+') as f:
		f.write('Last AUC\tLast AP\tLast Pr\tLast Re\tBest Epoch\tBest AUC\tBest AP\tBest Pr\tBest Re\n')

	# List of values here
	params = {
		'd_steps': [5,10],
		'lr': [0.01, 0.005, 0.0001]
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
	
