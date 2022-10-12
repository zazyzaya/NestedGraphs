import os 
import pickle
import sys

from dateutil.parser import isoparse
from joblib import Parallel, delayed
import torch 
from tqdm import tqdm

from .hasher import path_to_tensor
from .datastructures import FullGraph

# Globals 
JOBS = 17
SOURCE = '/mnt/raid0_24TB/datasets/NCR2/nested_optc/hosts/'

# Hyper parameters
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
    fields = line.split(',')
    ts,obj,act = fields[:3]

    ts = fmt_ts(ts)
    if ts == 0:
        return 

    if obj == 'PROCESS':
        if act != 'CREATE':
            return

        # When processes are 'OPENED' their 'parent' is the
        # source proc, and the target is the 'child'
        src_id,dst_id, pid,ppid, path,ppath = fields[3:]
        p_hr = fmt_p(pid,path); pp_hr = fmt_p(ppid,ppath)

        graph.add_edge(
            ts, src_id, dst_id,
            torch.zeros(DEPTH*4),
            torch.zeros(DEPTH*4),
            graph.NODE_TYPES[obj], 
            graph.NODE_TYPES[obj],
            EDGES[obj][act],
            human_src=p_hr, 
            human_dst=pp_hr
        )

    elif obj == 'FILE':  
        if act == 'RENAME':
            return 

        try:
            src_id,dst_id,path = fields[3:]
            path = path.strip()
            if not path:
                return 
        except ValueError:
            # I cannot figure out what's causing this. 
            # It seems to happen totally randomly on different files
            # hopefully this try-catch won't slow things down too much 
            return 



        # File -> Proc
        if act == 'READ':
            graph.add_edge(
                ts, dst_id, src_id,
                path_to_tensor(path, DEPTH),
                torch.zeros(DEPTH*4),
                graph.NODE_TYPES[obj],
                graph.NODE_TYPES['PROCESS'],
                EDGES[obj][act],
                #human_src=path.split('\\')[-1]
            )

        # Proc -> File
        else:
            graph.add_edge(
                ts, src_id, dst_id,
                torch.zeros(DEPTH*4),
                path_to_tensor(path, DEPTH),
                graph.NODE_TYPES['PROCESS'],
                graph.NODE_TYPES[obj],
                EDGES[obj][act],
                #human_dst=path.split('\\')[-1]
            )
    

def build_full_graph(i: int, tot: int, host: int, day: int, is_mal: bool, write=True):
    g = FullGraph(host)
    
    in_f = SOURCE+'Sept%d/sysclient%04d.csv' % (day,host)
    if not os.path.exists(in_f):
        return 

    print(in_f)
    prog = tqdm(desc='Lines parsed (%d/%d)' % (i+1,tot))
    with open(in_f) as f:
        line = f.readline()

        while(line):
            parse_line_full(g, line)
            line = f.readline()
            prog.update()

    prog.close()

    print("Finalizing graph")
    g.finalize(8) 

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

# Debug
if __name__ == '__main__':
    build_full_graph(0, 1, 559, 23, True, False)