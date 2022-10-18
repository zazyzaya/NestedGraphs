import glob 
import gzip 
import json 

import datetime as dt 
from joblib import Parallel, delayed
import torch 
from torch_geometric.data import Dataset
from tqdm import tqdm 

from .build_hostgraph import fmt_ts

NTYPES = {k:i for i,k in enumerate([
    'FILE', 'FLOW', 'HOST', 'MODULE', 'PROCESS', 'REGISTRY', 'SERVICE',
    'SHELL', 'TASK', 'THREAD', 'USER_SESSION'
])}

# A few duplicates, thowing them in a set first to get rid of them
ETYPES = {k:i for i,k in enumerate(set([
    'CREATE','DELETE','MODIFY','READ','RENAME','WRITE','MESSAGE','OPEN','START',
    'LOAD','CREATE','OPEN','TERMINATE','ADD','EDIT','REMOVE','COMMAND','START',
    'REMOTE_CREATE','TERMINATE','GRANT','INTERACTIVE','LOGIN','LOGOUT','RDP',
    'REMOTE','UNLOCK'
]))}

MAL = {
    23: [201,402,660,104,205,321,255,355,503,462,559,419,609,771,955,874,170]
}

DAY = 23
IN_FILES = glob.glob("/mnt/raid0_24TB/datasets/NCR2/ecar/evaluation/%dSep*/*/*.json*" % DAY)
OUT_DIR = 'inputs/Sept%d/unfiltered/' % DAY

class UnfilteredGraph(Dataset):
    def __init__(self, gid, feats):
        super().__init__()
        self.gid = gid 

        self.node_map = dict()
        self.human_readable = dict()
        self.nid = 0 

        self.one_hop = dict()

        self.feats = feats 
        self.x = []
        self.node_ts = []
        self.ntype = []

        self.final = False 

    def add_node(self, nid, ntype, ts, human_readable=None):
        if not nid in self.node_map:
            self.node_map[nid] = self.nid 
            self.nid += 1 

            self.x.append(torch.zeros(self.feats*2))
            self.node_ts.append(ts)
            self.ntype.append(ntype)

            if human_readable is not None:
                self.human_readable[nid] = human_readable

            return self.nid-1

        if human_readable is not None and not nid in self.human_readable:
            self.human_readable[nid] = human_readable

        return self.node_map[nid]

    
    def add_edge(self, src,dst, sx,dx, rel, ts, src_hr=None, dst_hr=None):
        src = self.add_node(src, sx, ts, src_hr)
        dst = self.add_node(dst, dx, ts, dst_hr)

        oh = self.one_hop.get(dst, set())
        oh.add(src)
        self.one_hop[dst] = oh 

        # Use the ThreaTrace method where feats are counts of types of 
        # in/outbound edges
        self.x[src][rel] += 1
        self.x[dst][self.feats+rel] += 1


    def add_src_feat(self, src,dst, sx,dx, rel, ts, src_hr=None, dst_hr=None):
        src = self.add_node(src, sx, ts, src_hr)
        self.x[src][rel] += 1

    def add_dst_feat(self, src,dst, sx,dx, rel, ts, src_hr=None, dst_hr=None):
        dst = self.add_node(dst, dx, ts, src_hr)
        self.x[dst][self.feats + rel] += 1


    def finalize(self):
        if self.final:
            return 
        self.final = True 

        self.x = torch.stack(self.x)
        self.node_ts = torch.tensor(self.node_ts)
        self.ntype = torch.tensor(self.ntype)

        # Store edges as csr matrix 
        ptr = [0] 
        idx = []
        for i in range(self.x.size(0)):
            idx_i = list(self.one_hop.get(i,set()))
            idx += idx_i
            ptr.append(ptr[-1]+len(idx_i))

        self.ptr = torch.tensor(ptr)
        self.idx = torch.tensor(idx)

    def get_one_hop(self, idx):
        st = self.ptr[idx]; end = self.ptr[idx+1]
        return self.idx[st:end]

    def to(self, device):
        self.x = self.x.to(device)
        self.ntype = self.ntype.to(device)
        self.ptr = self.ptr.to(device)
        self.idx = self.idx.to(device)

        return self 


def parse_line(line):
    obj = line['object']
    act = line['action']

    src = line['actorID']
    dst = line['objectID']

    # Principal inferred to be process unless specified otherwise
    src_x = NTYPES['PROCESS']
    dst_x = NTYPES[obj]
    rel = ETYPES[act]
    ts = fmt_ts(line['timestamp'])

    host_id = int(line['hostname'].replace('SysClient','').split('.',1)[0])

    # Some special cases where direction is reversed
    if (obj == 'FILE' and act == 'READ') or obj == 'REGISTRY':
        return host_id, (dst,src, dst_x,src_x, rel,ts)

    if obj == 'FLOW': 
        direction = line['properties']['direction']
        if direction == 'inbound': 
            return host_id, (dst,src, dst_x,src_x, rel,ts)

    return host_id, (src,dst, src_x,dst_x, rel,ts)

def parse_one(in_f):
    '''
    Each file has logs for 50 disjoint hosts, so it's safe to run them in parallel 
    '''
    graphs = dict()
    f = gzip.open(in_f, 'rb')

    for line in tqdm(f, desc=in_f.split('/')[-2]):
        line = json.loads(line)
        host,args = parse_line(line)

        g = graphs.get(host, UnfilteredGraph(host,len(ETYPES)))
        g.add_edge(*args)
        graphs[host] = g 


    print('Finalizing')
    for g in graphs.values():
        g.finalize() 

    print("Saving")
    for g in graphs.values():
        mal_str = 'mal/' if g.gid in MAL[DAY] else 'benign/'
        torch.save(g, OUT_DIR+mal_str+'%d_unfiltered.pkl' % g.gid)

def parse_all(jobs):
    Parallel(n_jobs=jobs, prefer='processes')(
        delayed(parse_one)(f) for f in IN_FILES
    )
