import glob 
import random 
from types import SimpleNamespace

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, RocCurveDisplay
from sklearn.metrics import confusion_matrix
import torch 
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm 

from models.sage import GraphSAGE
from preprocessing.build_unfiltered import ETYPES, NTYPES

torch.set_num_threads(16)

DAY = 23
HOME = 'inputs/Sept%d/unfiltered/' % DAY
LABELS = 'inputs/maybe_mal.txt'
PROCS = NTYPES['PROCESS']

DEVICE = 3
HYPERPARAMS = SimpleNamespace(
    hidden=32, layers=2, pool='mean',
    samples=128, batch_size=5000, 
    lr=0.001, epochs=500, tr_graphs=200
)

ce_loss = CrossEntropyLoss()
def step(g, model, minibatch):
    ys = g.ntype[minibatch]
    zs = model(g, minibatch)
    return ce_loss(zs, ys)

def train_one(hp, to_train=None):
    model = GraphSAGE(
        len(ETYPES)*2, hp.hidden, len(NTYPES),
        hp.layers, samples=hp.samples, 
        device=DEVICE, pool=hp.pool
    )

    # Generate list of nodes that need to be classified correctly before the
    # ensemble is finished being constructed
    if to_train is None:
        to_train = dict()
        graphs = glob.glob(HOME+'benign/*')
        
        for g in graphs:
            gid = int(g.split('/')[-1].split('_')[0])
            to_train[gid] = None 


    opt = Adam(model.parameters(), lr=hp.lr)
    gids = list(to_train.keys())
    random.shuffle(gids)

    model.train()
    epochs = 0
    while epochs < hp.epochs:
        for gid in gids:
            g = torch.load(HOME+'benign/%d_unfiltered.pkl' % gid)
            if to_train[gid] is None:
                to_train[gid] = torch.arange(g.x.size(0))

            # Fixed in newer objects, but don't want to rebuild
            # all of them just bc of this
            if type(g.ntype) == list:
                g.ntype = torch.tensor(g.ntype)

            g = g.to(DEVICE)
            batch = to_train[gid]
            for i in range((batch.size(0) // hp.batch_size)+1):
                minibatch = batch[i*hp.batch_size : (i+1)*hp.batch_size]

                opt.zero_grad()
                loss = step(g, model, minibatch)
                loss.backward()
                opt.step() 

                print('[%d] g%d, %0.2f (%d/%d)' % (epochs,gid,loss.item(),minibatch.max(),batch.size(0)))
                epochs += 1 

                if epochs > hp.epochs: 
                    break

            if epochs > hp.epochs: 
                    break
    
    return model 

@torch.no_grad()
def evaluate_one(model, to_train, r=1.5):
    no_more = []
    nodes_learned = 0
    tot_nodes = 0 
    for gid,nodes in tqdm(to_train.items()):
        g = torch.load(HOME+'benign/%d_unfiltered.pkl' % gid)
        if nodes is None:
            nodes = torch.arange(g.x.size(0))

        g = g.to(DEVICE)
        labels = g.ntype[nodes]
        guesses = torch.softmax(model(g, nodes), dim=1)
        topk = guesses.topk(2, dim=1)

        # If it guessed wrong, we still need to train with them
        correct_guess = topk.indices[:,0] == labels 
        ratio = topk.values[:,0] / topk.values[:,1]

        still_need = (~correct_guess).logical_or(ratio < r)
        still_need = still_need.nonzero().squeeze(-1)
        
        nodes_learned += nodes.size(0)-still_need.size(0)
        tot_nodes += nodes.size(0)
        
        if still_need.size(0):
            to_train[gid] = still_need 
        else:
            no_more.append(gid)

    for gid in no_more:
        del to_train[gid]
    
    print('Learned %d/%d nodes' % (nodes_learned, tot_nodes))
    return to_train

def evaluate(z,y, r=1.5):
    tk = torch.softmax(z,dim=1).topk(2,dim=1)
    
    correct_guess = tk.indices[:,0] == y
    ratio = tk.values[:,0] / tk.values[:,1]
    non_anom = correct_guess.logical_and(ratio > r)

    return non_anom 

@torch.no_grad()
def test():
    with open(LABELS, 'r') as f:
        labels = eval(f.read())

    gids = list(labels.keys())

    params, args, kwargs = torch.load('saved_models/threatrace.pt')
    kwargs['device'] = DEVICE
    model = GraphSAGE(*args, **kwargs)

    y_hat, ys = [],[]
    for i,gid in enumerate(gids):
        if not labels[gid][str(DAY)]:
            continue 

        g = torch.load(HOME+'mal/%s_unfiltered.pkl' % gid)
        if type(g.ntype) == list:
            g.ntype = torch.tensor(g.ntype)
        g = g.to(DEVICE)

        # Get labels 
        anoms = [] 
        for uuid_list in labels[gid][str(DAY)].values():
            for uuid in uuid_list:
                nid = g.node_map.get(uuid['uuid'])
                if nid:
                    anoms.append(nid)

        anoms = torch.tensor(anoms)
        y = torch.zeros(g.x.size(0))
        y[anoms] = 1 

        # Only look at processes
        y_h = torch.zeros(g.x.size(0), dtype=torch.bool, device=DEVICE)
        y_h[g.ntype != PROCS] = True 

        # Evaluate w ensemble of modules. If one thinks it's fine, then 
        # don't continue
        prog = tqdm(params, desc='(%d/%d)  %d unk' % (i+1,len(gids), (~y_h).sum()))
        for p in prog:
            model.load_state_dict(p)
            model.eval()
            batch = (~y_h).nonzero().squeeze(-1)
            
            z = model(g, batch)
            is_b = evaluate(z,g.ntype[batch])
            y_h[batch] = y_h[batch].logical_or(is_b)

            prog.desc = '(%d/%d)  %d unk' % (i+1,len(gids), (~y_h).sum())
        prog.close()

        # Only evaluate process scores for now
        procs = g.ntype == PROCS
        y = y[procs]
        y_h = y_h[procs]

        # Invert, bc list tells us if process is benign
        y_hat.append(~y_h); ys.append(y)
    
    y_hat = torch.cat(y_hat, dim=0).cpu()
    ys = torch.cat(ys, dim=0).cpu()

    stats = dict() 
    stats['Pr'] = precision_score(ys, y_hat, zero_division=1.)
    stats['Re'] = recall_score(ys, y_hat, zero_division=0.)
    stats['F1'] = f1_score(ys, y_hat)
    stats['Ac'] = accuracy_score(ys, y_hat)

    print(str(confusion_matrix(ys,y_hat)))
    for k,v in stats.items():
        print(k,v)

    return stats 

def train_all(hp):
    files = glob.glob(HOME+'benign/*.pkl')
    graphs = [int(f.split('/')[-1].split('_')[0]) for f in files]

    to_train = {k:None for k in graphs[:hp.tr_graphs]}
    models = []
    
    i=1
    while len(to_train):
        print("(%d)" % i)
        model = train_one(hp, to_train)
        to_train = evaluate_one(model, to_train)
        models.append(model.state_dict())
        i += 1

        torch.save(
            (models, model.args, model.kwargs), 
            'saved_models/threatrace.pt'
        )

if __name__ == '__main__':
    train_all(HYPERPARAMS)
    test()