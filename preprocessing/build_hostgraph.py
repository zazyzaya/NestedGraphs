import sys 

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
MOD_DEPTH = 4 # It's very rarely > 3
DEPTH = 16

EDGES = {
    'PROCESS': {
        'CREATE': 0, 'OPEN': 1
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
fmt_ts = lambda x : isoparse(x).timestamp()
fmt_p = lambda pid,proc : pid.strip() + ':' + proc.strip().upper().split('\\\\')[-1].replace("'",'')

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
        # For now just add process.create or open events
        if act not in ['CREATE', 'OPEN']:
            return 

        # When processes are 'OPENED' their 'parent' is the
        # source proc, and the target is the 'child'
        pid, ppid, path, ppath = feats
        graph.add_edge(
            ts, fmt_p(pid,path), fmt_p(ppid,ppath), 
            proc_feats(path, PROC_DEPTH), 
            proc_feats(ppath, PROC_DEPTH),
            EDGES['PROCESS'][act],
            nodelist
        )

    elif obj == 'FILE':
        pid, ppid, path, p_img = feats[:4]

        if act == 'RENAME': 
            print(feats)

        if act != 'READ':
            nodelist.add_file(ts, fmt_p(pid,p_img), file_feats(path, act, FILE_DEPTH))

    elif obj == 'REGISTRY': 
        pid, ppid, key, _, p_img = feats[:5]
        nodelist.add_reg(ts, fmt_p(pid,p_img), reg_feats(key, act, REG_DEPTH))

    '''
    elif obj == 'MODULE': 
        pid, ppid, p_img, mod = feats[:4]

        if mod.endswith('.dll'):
            nodelist.add_mod(ts, fmt_p(pid,p_img), mod_feats(mod, MOD_DEPTH))
    '''

def build_graph(host: int, day: int):
    g = HostGraph(host)
    nl = NodeList()
    prog = tqdm(desc='Lines parsed')
    
    with open(SOURCE+'Sept%d/sysclient%04d.csv' % (day,host)) as f:
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
    

def build_graphs(hosts, day):
    '''
    Given a sequence of hosts, build graphs for them in parallel
    '''
    return Parallel(n_jobs=min(JOBS,len(hosts)), prefer='processes')(
        delayed(build_graph)(h, day) for h in hosts
    )


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
        if act == 'TERMINATE':
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

        if act != 'READ':
            graph.add_edge(
                ts, fmt_p(pid,p_img), path,
                path_to_tensor(p_img, DEPTH),
                path_to_tensor(path, DEPTH),
                graph.NODE_TYPES['PROCESS'],
                graph.NODE_TYPES[obj],
                EDGES[obj][act]
            )
        # Otherwise, edge direction is F -> P
        else: 
            graph.add_edge(
                ts, path, fmt_p(pid,p_img),
                path_to_tensor(path, DEPTH),
                path_to_tensor(p_img, DEPTH),
                graph.NODE_TYPES[obj],
                graph.NODE_TYPES['PROCESS'],
                EDGES[obj][act]
            )

    elif obj == 'REGISTRY': 
        pid, ppid, key, _, p_img = feats[:5]
        graph.add_edge(
            ts, fmt_p(pid,p_img), key, 
            path_to_tensor(p_img, DEPTH),
            path_to_tensor(key, DEPTH), 
            graph.NODE_TYPES['PROCESS'],
            graph.NODE_TYPES[obj],
            EDGES[obj][act]
        )

def build_full_graph(host: int, day: int):
    g = FullGraph(host)
    prog = tqdm(desc='Lines parsed')
    
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

def build_full_graphs(hosts, day):
    '''
    Given a sequence of hosts, build graphs for them in parallel
    '''
    return Parallel(n_jobs=min(JOBS,len(hosts)), prefer='processes')(
        delayed(build_full_graph)(h, day) for h in hosts
    )

if __name__ == '__main__':
    build_graph(201,23)