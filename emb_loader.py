import glob 
import pickle 
import torch 

from graph_utils import propagate_labels
from tqdm import tqdm 

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/'
EMB_NAME = 'tgat_het_emb'
GRAPH_NAME = 'full_graph'

def save_labeled_vecs(day=23):
    b_embs = glob.glob(HOME+'Sept%d/benign/%s*' % (day, EMB_NAME))
    m_embs = glob.glob(HOME+'Sept%d/mal/%s*' % (day, EMB_NAME))

    bs = []
    for b in tqdm(b_embs):
        with open(b.replace(EMB_NAME, GRAPH_NAME), 'rb') as f:
            g = pickle.load(f)

        mask = torch.tensor(g.ntypes) == 0
        bs.append(torch.load(b)[mask])

    
    bs = torch.cat([b for b in bs], dim=0)
    torch.save(bs, HOME+'Sept%d/benign/all_proc_embs.pkl' % day)
    print('Saved',bs.size(0),'benign process embeddings')

    ms = []
    y = []
    for m in tqdm(m_embs):
        with open(m.replace(EMB_NAME, GRAPH_NAME), 'rb') as f:
            g = pickle.load(f)
        
        proc_mask = torch.tensor(g.ntypes)==0

        ms.append(torch.load(m)[proc_mask])
        y.append(propagate_labels(g,day)[proc_mask])

    ms = torch.cat([m for m in ms])
    torch.save(ms, HOME+'Sept%d/mal/all_proc_embs.pkl' % day)
    print('Saved',ms.size(0),'mixed-malicious process embeddings')

    y = torch.cat(y).clamp(0,1)
    torch.save(y, HOME+'Sept%d/mal/ys.pkl' % day)

if __name__ == '__main__':
    save_labeled_vecs()