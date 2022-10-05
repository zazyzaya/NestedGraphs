import os 
import pickle
import sys

from dateutil.parser import isoparse
from joblib import Parallel, delayed
from tqdm import tqdm

from .hasher import path_to_tensor
from .datastructures import FullGraph

# Globals 
JOBS = 17
SOURCE = '/mnt/raid0_24TB/datasets/NCR2/nested_optc/hosts/'

# Hyper parameters
PROC_DEPTH = 16
FILE_DEPTH = 16
REG_DEPTH = 16
DEPTH = 16

EDGES = {
    'PROCESS': {
        'CREATE': 0, 'OPEN': 1, # Open is unused.. produces too much clutter
        'TERMINATE': 2
    },
    'FILE': {
        'CREATE': 3, 'MODIFY': 4, 
        'READ': 5, 'WRITE': 6, 
        'DELETE': 7,
        'RENAME': None
    },
    # No registry read events are captured by eCAR... 
    # is it even worth tracking these? 
    'REGISTRY': {
        'ADD': 8, 'EDIT': 9, 'REMOVE': 10
    },
    'MODULE': {
        'LOAD': 11
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
        if act != 'CREATE':
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

        # File -> Proc
        if act == 'READ':
            graph.add_edge(
                ts, path, fmt_p(pid,p_img),
                path_to_tensor(path, DEPTH),
                path_to_tensor(p_img, DEPTH),
                graph.NODE_TYPES[obj],
                graph.NODE_TYPES['PROCESS'],
                EDGES[obj][act],
            )

        # Proc -> File
        else:
            graph.add_edge(
                ts, fmt_p(pid,p_img), path,
                path_to_tensor(p_img, DEPTH),
                path_to_tensor(path, DEPTH),
                graph.NODE_TYPES['PROCESS'],
                graph.NODE_TYPES[obj],
                EDGES[obj][act],
            )
    

    elif obj == 'REGISTRY': 
        pid, ppid, key, _, p_img = feats[:5]
        graph.add_edge(
            ts, fmt_p(pid,p_img), key, 
            path_to_tensor(p_img, DEPTH),
            path_to_tensor(key, DEPTH, reverse=True), 
            graph.NODE_TYPES['PROCESS'],
            graph.NODE_TYPES[obj],
            EDGES[obj][act],
            bidirectional=True
        )

    elif obj == 'MODULE': 
        pid, _, mod, p_img = feats

        # Always a load event (mod -> proc)
        # kind of unique in that they always point to procs and have
        # no parents. Don't want to treat them as regular edges
        # Have plans of using them as process features when aggregated?
        graph.add_module(
            ts, fmt_p(pid,p_img), mod, 
            path_to_tensor(p_img, DEPTH), 
            path_to_tensor(mod, DEPTH)
        )

def build_full_graph(i: int, tot: int, host: int, day: int, is_mal: bool, write=True):
    g = FullGraph(host)
    prog = tqdm(desc='Lines parsed (%d/%d)' % (i+1,tot))
    
    in_f = SOURCE+'Sept%d/sysclient%04d.csv' % (day,host)
    if not os.path.exists(in_f):
        prog.close()
        return 

    with open(in_f) as f:
        line = f.readline()
        while(line):
            parse_line_full(g, line)
            line = f.readline()
            prog.update()


    prog.close()

    print("Finalizing graph")
    g.finalize(12) 

    if write:
        out_f = 'inputs/Sept%d/benign/full_graph%d.pkl' % (day,host)
        if is_mal:
            out_f = out_f.replace('benign', 'mal')
            
        with open(out_f, 'wb+') as f:
            pickle.dump(g, f)

    return g

def build_full_graphs(hosts, day, jobs=JOBS, is_mal=False):
    '''
    Given a sequence of hosts, build graphs for them in parallel
    '''
    return Parallel(n_jobs=min(jobs, len(hosts)), prefer='processes')(
        delayed(build_full_graph)(i, len(hosts), h, day, is_mal) for i,h in enumerate(hosts)
    )