import glob 
import pickle
import random 
from types import SimpleNamespace

from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score, \
    recall_score, precision_score, f1_score, accuracy_score
import torch 
import torch.nn.functional as F
from torch import nn 
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from train_multigpu import HOME
from utils.graph_utils import get_edge_index

ALPHA = 0.8
BETA = 15
GAMMA = 0.1

WORKERS = 16
DEVICE = 3
HYPERPARAMS = SimpleNamespace(
    hidden=512, layers=3, latent=64, 
    d_lr=0.0001, g_lr=0.001, 
    epochs=100
)


class DiscGCN(nn.Module):
    def __init__(self, in_dim, hidden, layers, device=torch.device('cpu')):
        super().__init__()
        self.args = (in_dim, hidden, layers)
        self.kwargs = dict(device=device)
        
        self.drop = nn.Dropout()
        self.nets = nn.ModuleList(
            [GCNConv(in_dim, hidden)] + 
            [GCNConv(hidden, hidden) for _ in range(layers-2)] +
            [GCNConv(hidden, 1)]
        )

        # God damn it, PyG. Fix your API. 
        # Why can't I initialize these in the GPU?
        self.nets.to(device)

    def forward(self, x, ei):
        for net in self.nets[:-1]:
            x = self.drop(torch.relu(net(x, ei)))

        return torch.sigmoid(self.nets[-1](x,ei))

class GenGCN(DiscGCN):
    def __init__(self, in_dim, latent_dim, hidden, out, layers, device=torch.device('cpu')):
        super().__init__(in_dim, hidden, layers, device)
        self.args = (in_dim, latent_dim, hidden, out, layers)
        self.kwargs = dict(device=device)

        self.static_net = nn.Sequential(
            nn.Linear(in_dim+latent_dim, hidden, device=device), 
            nn.ReLU(),
            nn.Linear(hidden, in_dim, device=device),
            nn.ReLU()
        )
        self.nets[-1] = GCNConv(hidden, out)
        self.nets[-1].to(device)

        self.device = device
        self.latent_dim = latent_dim

    def forward(self, x, ei):
        # Incorporate noise 
        static = torch.rand(
            (x.size(0),self.latent_dim), 
            device=self.device
        )
        x = torch.cat([x,static], dim=1)
        x = self.static_net(x)

        for net in self.nets[:-1]:
            x = self.drop(torch.relu(net(x,ei)))

        # No final activation
        return self.nets[-1](x,ei)



def disc_step(disc, gen, graph, embs, procs):
    ei = get_edge_index(graph)
    
    real = disc(embs, ei)[procs]
    fake = disc(
        gen(graph.x, ei), ei 
    )[procs]

    return (-torch.log(real+1e-9) - GAMMA * torch.log(1-fake+1e-9)).mean()

def gen_step(disc, gen, graph, embs, procs):
    ei = get_edge_index(graph)

    fake = gen(graph.x, ei)
    mu = fake.mean(dim=0)

    disp_loss = BETA * (1 / ((fake-mu).pow(2)).mean())
    recon_loss = torch.log((ALPHA - disc(fake, ei)[procs]).abs()).mean()

    return recon_loss + disp_loss

def train(hp):
    torch.set_num_threads(WORKERS)
    graphs = glob.glob(HOME+'full_graph*')
    
    with open(graphs[0],'rb') as f:
        g = pickle.load(f).to(DEVICE)

    disc = DiscGCN(
        128, hp.hidden, hp.layers, device=DEVICE
    )
    d_opt = Adam(disc.parameters(), lr=hp.d_lr)

    gen = GenGCN(
        g.x.size(1), hp.latent, hp.hidden, 128,
        hp.layers, device=DEVICE
    )
    g_opt = Adam(gen.parameters(), lr=hp.g_lr)
    

    for e in range(hp.epochs):
        random.shuffle(graphs)

        for i,g_file in enumerate(graphs):
            with open(g_file,'rb') as f:
                g = pickle.load(f).to(DEVICE)
            
            embs = torch.load(
                g_file.replace('full_', 'tgat_emb_clms')
            )
            zs = F.normalize(embs['zs'].to(DEVICE))
            procs=embs['proc_mask'].to(DEVICE)

            # Disc step 
            d_opt.zero_grad()
            disc.train(); gen.eval()
            gen.requires_grad = False 
            disc.requires_grad = True 
            
            d_loss = disc_step(disc, gen, g, zs, procs)
            d_loss.backward()
            d_opt.step() 

            # Gen step 
            g_opt.zero_grad()
            disc.eval(); gen.train() 
            gen.requires_grad = True 
            disc.requires_grad = False 

            g_loss = gen_step(disc, gen, g, embs, procs)
            g_loss.backward() 
            g_opt.step() 

            print(
                '[%d-%d] D Loss: %0.4f  G Loss: %0.4f' % 
                (e,i+1,d_loss.item(), g_loss.item())
            )
            '''
            torch.save(
                (
                    model.state_dict(), 
                    model.args, 
                    model.kwargs
                ), 'saved_models/anom.pkl'
            )
            '''

            # Try to save some memory 
            del g,zs
        
        test(disc)

@torch.no_grad()
def test(model, thresh=None):
    model.eval()
    preds = []; ys = []
    graphs = glob.glob(HOME.replace('benign', 'mal')+'full_graph*')

    for g_file in graphs:
        with open(g_file, 'rb') as f:
            g = pickle.load(f).to(DEVICE)

        embs = torch.load(
            g_file.replace('full_', 'tgat_emb_clms')
        )
        
        zs = embs['zs'].to(DEVICE)
        procs = embs['proc_mask'].to(DEVICE)
        ys.append(embs['y'].to('cpu'))
        preds.append(model(zs, get_edge_index(g))[procs])

    # Higher preds -> more anomalous now
    preds = 1-torch.sigmoid(torch.cat(preds)).to('cpu')
    ys = torch.cat(ys).clamp(0,1).to('cpu')

    if thresh is None:
        thresh = preds.quantile(0.99)

    y_hat = torch.zeros(preds.size())
    y_hat[preds > thresh] = 1 

    stats = dict() 
    stats['Pr'] = precision_score(ys, y_hat)
    stats['Re'] = recall_score(ys, y_hat)
    stats['F1'] = f1_score(ys, y_hat)
    stats['Ac'] = accuracy_score(ys, y_hat)
    stats['AUC'] = auc_score(ys, preds)
    stats['AP'] = ap_score(ys, preds)

    for k,v in stats.items():
        print(k,v)

    return stats 

if __name__ == '__main__':
    train(HYPERPARAMS)