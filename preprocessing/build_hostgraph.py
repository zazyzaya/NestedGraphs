from dateutil.parser import isoparse
from joblib import Parallel, delayed

from tqdm import tqdm

from hasher import proc_feats, file_feats, reg_feats, mod_feats 
from datastructures import HostGraph

# Globals 
JOBS = 16
SOURCE = '/mnt/raid0_24TB/datasets/NCR2/nested_optc/hosts'

# Hyper parameters
PROC_DEPTH = 8
FILE_DEPTH = 8
REG_DEPTH = 4
MOD_DEPTH = 3 # It's very rarely > 3

# Converts from ISO timestamp to UTC time since epoch
fmt_ts = lambda x : isoparse(x).timestamp()

def parse_line(graph: HostGraph, line: str) -> None:
    '''
    Adds process node to the graph if needed, along with imgpath feature

        Args: 
            data (HostGraph): data object containing the graph
            line (str): line from hostlog csv file
    '''
    fields = line.split(',', 3)
    ts, obj, act = fields[:3]
    feats = fields[3].replace('"','').split(',')

    if obj == 'PROCESS':
        # For now just add process.create events
        if act != 'CREATE':
            return 

        pid, ppid, path = feats[:-1]
        graph.add_edge(ts, pid, ppid, proc_feats(path, PROC_DEPTH))

    elif obj == 'FILE':
        pid, ppid, path = feats[:-1]
        graph.add_file(ts, pid, file_feats(path, act, FILE_DEPTH))

    elif obj == 'REGISTRY': 
        pid, ppid, key = feats[:-2]
        graph.add_reg(ts, pid, reg_feats(key, act, REG_DEPTH))

    elif obj == 'MODULE': 
        pid, ppid, _, mod = feats 
        graph.add_mod(ts, pid, mod_feats(mod, MOD_DEPTH))


def build_graph(host: int) -> HostGraph:
    g = HostGraph(host)
    prog = tqdm(desc='Lines parsed')
    
    with open(SOURCE+'sysclient%04d.csv' % host) as f:
        line = f.readline()

        while(line):
            parse_line(g, line)
            line = f.readline()
            tqdm.update()

    g.finalize() 
    return g
    

def build_graphs(hosts):
    '''
    Given a sequence of hosts, build graphs for them in parallel
    '''
    return Parallel(n_jobs=JOBS, prefer='processes')(
        delayed(build_graph)(h) for h in hosts
    )