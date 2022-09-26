import glob
import os
import pickle 
import random 

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score, \
    recall_score, precision_score, f1_score, accuracy_score, RocCurveDisplay
import torch 
from torch import nn 
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm 

from models.tgat import TGAT 
from models.detector import SimpleDetector
from train_multigpu import fair_scheduler, HOME, HYPERPARAMS as TGAT_HP
from train_anomdetector import HYPERPARAMS as ANOM_HP

DEVICE = 3
P_THREADS = 8

class Pipeline(nn.Module):
    '''
    Wrapper to train TGAT in conjunction with detector
    '''
    def __init__(self, tgat, detector):
        super().__init__() 

        self.tgat = tgat
        self.detector = detector 
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, g, batch):
        zs = self.tgat(g, batch=torch.arange(g.x.size(0), device=g.x.device))
        
        src,dst = [],[]
        for nid in batch:
            st = g.csr_ptr[nid]
            en = g.csr_ptr[nid+1]

            dst.append(g.edge_index[st:en])
            src.append(nid.repeat(en-st))

        src = torch.cat(src).long()
        dst = torch.cat(dst).long()
        neg_d = torch.randint(
            0, g.x.size(0), 
            dst.size(), device=zs.device
        )

        pos = self.detector(zs[src], zs[dst])
        neg = self.detector(zs[src], zs[neg_d])

        p_labels = torch.full(pos.size(), 1., device=src.device)
        n_labels = torch.full(neg.size(), 0., device=src.device)

        return self.bce(
            torch.cat([pos, neg], dim=0),
            torch.cat([p_labels, n_labels], dim=0)
        )

    def predict(self, g, mask, zs=None):
        results = torch.zeros((mask.size(0),1), device=g.x.device)

        src,dst = [],[]
        idx_map = []; i=0
        for nid in mask:
            st = g.csr_ptr[nid.item()]
            en = g.csr_ptr[nid.item()+1]

            dst.append(g.edge_index[st:en])
            src.append(nid.repeat(en-st))
            idx_map.append(torch.tensor(i, device=g.x.device).repeat(en-st))
            i += 1

        if zs is None:
            zs = self.tgat(g, batch=torch.arange(g.x.size(0))).detach()

        src = torch.cat(src).long()
        dst = torch.cat(dst).long()
        idx = torch.cat(idx_map).long()
        preds = self.detector(zs[src],zs[dst])

        # Get the neighbor with maximum suspicion to use as the 
        # score for this node
        return results.index_reduce_(0, idx, preds, 'amin', include_self=False)


def proc_job(t_hp, a_hp):
    # Sets number of threads used by this worker
    torch.set_num_threads(P_THREADS)
    graphs = glob.glob(HOME+'full_graph*')
    
    print("Loading graph")
    with open(graphs[0],'rb') as f:
        g = pickle.load(f)

    tgat = TGAT(
        g.x.size(1), g.edge_feat_dim, 
        t_hp.tsize, t_hp.hidden, t_hp.emb_size, 
        t_hp.layers, t_hp.heads,
        neighborhood_size=t_hp.nsize//4,
        device=DEVICE
    )
    detector = SimpleDetector(
        t_hp.emb_size, a_hp.hidden, a_hp.layers, 
        device=DEVICE
    )
    model = Pipeline(tgat, detector)

    opt = Adam(model.parameters(), lr=t_hp.lr)
    loss_item = torch.tensor([float('nan')])
    for e in range(t_hp.epochs):
        model.train()

        random.shuffle(graphs)
        prog = tqdm(
            total=len(graphs), 
            desc='[%d-%d] Loss: %0.4f' % (e,0,loss_item)
        )

        for i,g_file in enumerate(graphs):
            with open(g_file,'rb') as f:
                g = pickle.load(f).to(DEVICE)

            # Get this processes batch of jobs. In this case, 
            # nids of nodes that represent processes (x_n = [1,0,0,...,0])
            procs = (g.x[:,0] == 1).nonzero().squeeze(-1)

            #my_batch = torch.tensor(fair_scheduler(world_size, costs)[rank]).to(DEVICES[rank]).long()
            opt.zero_grad()
            loss = model.forward(g, procs)
            loss.backward()
            opt.step() 
            
            prog.desc = '[%d-%d] Loss: %0.4f' % (e,i+1,loss.item())
            prog.update()

            with open('log.txt', 'a') as f:
                f.write('%f\n' % loss.item())

            torch.save(
                model.state_dict(), 
                'saved_models/pipeline.pkl'
            )

            # Try to save some memory 
            loss_item = loss.item()
            del g,loss
            torch.cuda.empty_cache()

        if e%10 == 9:    
            stats = test_per_cc(model)
            with open('output.txt', 'a') as f:
                for k,v in stats.items():
                    f.write('%s %f\n' % (k,v))

        prog.close() 
        


@torch.no_grad()
def test_one_per_cc(model, g, procs, ccs):
    results = model.predict(g, procs)
    
    # Take the average(?) score of the connected component
    cc_results = torch.zeros((len(ccs), 1), device=procs.device)
    for i in range(len(ccs)):
        cc_results[i] = results[ccs[i]].min()

    return cc_results

def test_per_cc(model, thresh=None):
    preds = []; ys = []
    graphs = glob.glob(HOME.replace('benign', 'mal')+'full_graph*')

    for g_file in graphs:
        with open(g_file, 'rb') as f:
            g = pickle.load(f).to(DEVICE)

        embs = torch.load(
            g_file.replace('full_', 'tgat_emb_clms')
        )
        
        procs = embs['proc_mask'].to(DEVICE)
        node_ys = embs['y']
        ccs = embs['ccs']

        ys.append(torch.stack([node_ys[cc].max() for cc in ccs]))
        preds.append(test_one_per_cc(model, g, procs, ccs))

    # Higher preds -> more anomalous now
    preds = 1-torch.sigmoid(torch.cat(preds)).to('cpu')
    ys = torch.cat(ys).to('cpu')

    if thresh is None:
        thresh = preds.quantile(0.99)
    else:
        thresh = 1-torch.sigmoid(thresh).cpu()

    y_hat = torch.zeros(preds.size())
    y_hat[preds > thresh] = 1 

    ys = ys.clamp(0,1)
    
    stats = dict() 
    stats['Pr'] = precision_score(ys, y_hat)
    stats['Re'] = recall_score(ys, y_hat)
    stats['F1'] = f1_score(ys, y_hat)
    stats['Ac'] = accuracy_score(ys, y_hat)
    stats['AUC'] = auc_score(ys, preds)
    stats['AP'] = ap_score(ys, preds)

    print("%d samples; %d anomalies" % (ys.size(0), ys.sum().item()))
    for k,v in stats.items():
        print(k,v)

    RocCurveDisplay.from_predictions(ys, preds)
    plt.savefig('roc_curve.png')
    return stats 


if __name__ == '__main__':
    with open('log.txt', 'w+'):
        pass 
    with open('output.txt', 'w+'):
        pass 

    proc_job(TGAT_HP,ANOM_HP)