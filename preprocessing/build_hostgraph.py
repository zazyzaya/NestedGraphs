import sys
from unicodedata import bidirectional 

from dateutil.parser import isoparse
from joblib import Parallel, delayed
from tqdm import tqdm

from .hasher import proc_feats, file_feats, reg_feats, path_to_tensor
from .datastructures import HostGraph, FullGraph, NodeList

# Globals 
JOBS = 8
SOURCE = '/mnt/raid0_24TB/datasets/NCR2/nested_optc/hosts/'

# Hyper parameters
PROC_DEPTH = 16
FILE_DEPTH = 16
REG_DEPTH = 16
DEPTH = 16

EDGES = {
    'PROCESS': {
        'CREATE': 0, 'OPEN': 1 # Open is unused.. produces too much clutter
    },
    'FILE': {
        'CREATE': 2, 'MODIFY': 3, 
        'READ': 4, 'WRITE': 5, 
        'DELETE': 6,
        'RENAME': None
    },
    'REGISTRY': {
        'ADD': 7, 'EDIT': 8, 'REMOVE': 9
    }
}

# Converts from ISO timestamp to UTC time since epoch
# Also, make the number a bit smaller; torch really hates saving
# all those sig figs later
fmt_ts = lambda x : isoparse(x).timestamp() - 1569000000
fmt_p = lambda pid,proc : pid.strip() + ':' + proc.strip().upper().split('\\\\')[-1].replace("'",'')

def parse_line_full(graph: FullGraph, line: str) -> None:
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
        if act not in ['CREATE', 'OPEN']:
            return 

        # When processes are 'OPENED' their 'parent' is the
        # source proc, and the target is the 'child'
        pid, ppid, path, ppath = feats
        graph.add_edge(
            ts, fmt_p(ppid,ppath), fmt_p(pid,path),
            path_to_tensor(ppath, DEPTH),
            path_to_tensor(path, DEPTH), 
            graph.NODE_TYPES[obj], 
            graph.NODE_TYPES[obj],
            EDGES[obj][act],
        )

    elif obj == 'FILE':  
        pid, ppid, path, p_img, new_path = feats[:5]

        if act == 'RENAME': 
            graph.update_uuid(
                path, new_path
            )
            return 

        graph.add_edge(
            ts, fmt_p(pid,p_img), path,
            path_to_tensor(p_img, DEPTH),
            path_to_tensor(path, DEPTH),
            graph.NODE_TYPES['PROCESS'],
            graph.NODE_TYPES[obj],
            EDGES[obj][act],
            bidirectional=True
        )
    

    elif obj == 'REGISTRY': 
        pid, ppid, key, _, p_img = feats[:5]
        graph.add_edge(
            ts, fmt_p(pid,p_img), key, 
            path_to_tensor(p_img, DEPTH),
            path_to_tensor(key, DEPTH), 
            graph.NODE_TYPES['PROCESS'],
            graph.NODE_TYPES[obj],
            EDGES[obj][act],
            bidirectional=True
        )

def build_full_graph(i: int, tot: int, host: int, day: int):
    g = FullGraph(host)
    prog = tqdm(desc='Lines parsed (%d/%d)' % (i+1,tot))
    
    with open(SOURCE+'Sept%d/sysclient%04d.csv' % (day,host)) as f:
        line = f.readline()
        while(line):
            parse_line_full(g, line)
            line = f.readline()
            prog.update()


    prog.close()

    print("Finalizing graph")
    g.finalize(9) 
    return g

def build_full_graphs(hosts, day, jobs=JOBS):
    '''
    Given a sequence of hosts, build graphs for them in parallel
    '''
    return Parallel(n_jobs=min(jobs, len(hosts)), prefer='processes')(
        delayed(build_full_graph)(i, len(hosts), h, day) for i,h in enumerate(hosts)
    )