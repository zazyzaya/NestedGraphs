from dateutil.parser import isoparse
from joblib import Parallel, delayed

from tqdm import tqdm

from .hasher import proc_feats, file_feats, reg_feats, mod_feats 
from .datastructures import HostGraph, NodeList

# Globals 
JOBS = 8
SOURCE = '/mnt/raid0_24TB/datasets/NCR2/nested_optc/hosts/'

# Hyper parameters
PROC_DEPTH = 8
FILE_DEPTH = 8
REG_DEPTH = 8
MOD_DEPTH = 4 # It's very rarely > 3

# Converts from ISO timestamp to UTC time since epoch
fmt_ts = lambda x : isoparse(x).timestamp()

def parse_line(graph: HostGraph, nodelist: NodeList, line: str) -> None:
    '''
    Adds process node to the graph if needed, along with imgpath feature

        Args: 
            data (HostGraph): data object containing the graph
            line (str): line from hostlog csv file
    '''
    fields = line.split(',', 3)
    ts, obj, act = fields[:3]
    feats = fields[3][2:-3].split(',')

    ts = fmt_ts(ts)
    if ts == 0:
        return 

    if obj == 'PROCESS':
        # For now just add process.create events
        if act != 'CREATE':
            return 

        pid, ppid, path, ppath = feats
        graph.add_edge(
            ts, pid, ppid, 
            proc_feats(path, PROC_DEPTH), 
            proc_feats(ppath, PROC_DEPTH),
            nodelist
        )

    elif obj == 'FILE':
        pid, ppid, path = feats[:3]
        nodelist.add_file(ts, pid, file_feats(path, act, FILE_DEPTH))

    elif obj == 'REGISTRY': 
        pid, ppid, key = feats[:3]
        nodelist.add_reg(ts, pid, reg_feats(key, act, REG_DEPTH))

    elif obj == 'MODULE': 
        pid, ppid, _, mod = feats 
        nodelist.add_mod(ts, pid, mod_feats(mod, MOD_DEPTH))


def build_graph(host: int):
    g = HostGraph(host)
    nl = NodeList()
    prog = tqdm(desc='Lines parsed')
    
    with open(SOURCE+'sysclient%04d.csv' % host) as f:
        line = f.readline()

        while(line):
            parse_line(g, nl, line)
            line = f.readline()
            prog.update()

    prog.close()

    print("Finalizing graph")
    g.finalize() 
    print("Finalizing nodes")
    nl.finalize()
    return g, nl
    

def build_graphs(hosts):
    '''
    Given a sequence of hosts, build graphs for them in parallel
    '''
    return Parallel(n_jobs=JOBS, prefer='processes')(
        delayed(build_graph)(h) for h in hosts
    )