import glob
import json 
import pickle

from joblib import Parallel, delayed
from tqdm import tqdm 

from .hasher import path_to_tensor
from .datastructures import FullGraph


HOME = '/mnt/raid0_24TB/datasets/TC/Engagement3/data/fivedirection/'

EDGES = {
    'EVENT_UNLINK':0, 'EVENT_FCNTL':1, 'EVENT_SIGNAL':2, 
    'EVENT_RECVFROM':3, 'EVENT_CREATE_OBJECT':4, 
    'EVENT_SENDTO':5, 'EVENT_FORK':6, 'EVENT_UPDATE':7, 
    'EVENT_OTHER':8, 'EVENT_OPEN':9, 'EVENT_CHECK_FILE_ATTRIBUTES':10, 
    'EVENT_MODIFY_FILE_ATTRIBUTES':13,'EVENT_ACCEPT':14,
    'EVENT_WRITE':15, 'EVENT_BIND':16,
    'EVENT_EXECUTE':17, 'EVENT_READ':18, 'EVENT_CLOSE':19, 
    'EVENT_RECVMSG':20, 'EVENT_RENAME':21, 'EVENT_LINK':22, 
    'EVENT_SENDMSG':23, 'EVENT_CONNECT':24
}

PROC=0
REG=1
FILE=2 

DEPTH = 8

def build_db(i, tot, fname):
    etypes = set() 
    uuids = dict() 

    f = open(fname, 'r')
    line = f.readline() 

    prog = tqdm(desc="(%d/%d)" % (i,tot))
    while(line):
        line = json.loads(line)['datum']
        ltype = next(iter(line))

        if ltype.endswith('Event'):
            etypes.add(line[ltype]['type'])

        elif ltype.endswith('RegistryKeyObject'):
            line = line[ltype]
            uuids[line['uuid']] = (REG,line['key'])

        elif ltype.endswith('Subject'):
            line = line[ltype]
            if line['type'] == 'SUBJECT_PROCESS': 
                if line.get('cmdLine'):
                    cmd = line['cmdLine']['string']
                else:
                    cmd = ''

                uuids[line['uuid']] = (PROC,line['uuid'],line['cid'],cmd)

            elif line['type'] == 'SUBJECT_THREAD':
                parent_d = line['parentSubject']
                parent_k = next(iter(parent_d))
                parent = parent_d[parent_k]

                # Point to parent process
                if uuids.get(parent):
                    uuids[line['uuid']] = uuids[parent]

        prog.update()
        line = f.readline()

    f.close()
    return etypes, uuids 


def build_all_dbs():
    files = glob.glob(HOME+'*-2.json*')
    files = [f for f in files if not f.endswith('.tar.gz')]

    results = Parallel(n_jobs=8, prefer='processes')(
        delayed(build_db)(i, len(files), f) for i,f in enumerate(files)
    )

    etypes, dbs = zip(*results)
    
    etypes = set().union(*etypes)
    print(etypes)

    db = {k:v for d in dbs for k,v in d.items()}
    with open('preprocessing/e3_ids.pkl', 'wb+') as f:
        pickle.dump(db, f)

    with open('preprocessing/e3_edges.txt', 'w+') as f:
        for e in etypes:
            f.write(e+'\n')

fmt_time = lambda x : (x // 1e6) - 1523318400000
def parse_line(line,g,ids):
    ltype = next(iter(line))
    if not ltype.endswith('Event'):
        return 

    line = line[ltype]
    
    etype = EDGES.get(line['type'])
    if etype is None:
        return 

    src = list(line['subject'].values())[0]

    # Sometimes EVENT_OTHER is a non-edge. Check here
    if line['predicateObject']:
        dst = list(line['predicateObject'].values())[0]
    else:
        return
        

    src = ids.get(src)
    if src is None:
        return 

    _,uuid,pid,exe = src 
    if dst_info := ids.get(dst):
        ntype = dst_info[0]

        if ntype == REG: 
            g.add_edge(
                fmt_time(line['timestampNanos']), 
                uuid,dst, 
                path_to_tensor(exe, DEPTH), 
                path_to_tensor(dst_info[1], DEPTH, reverse=True), 
                PROC, REG, etype,
                human_src=':'.join([str(pid),exe]),
                human_dst=dst_info[1]
            )

        elif ntype == PROC: 
            g.add_edge(
                fmt_time(line['timestampNanos']), 
                uuid,dst_info[1], 
                path_to_tensor(exe, DEPTH), 
                path_to_tensor(dst_info[-1], DEPTH), 
                PROC, PROC, etype,
                human_src=':'.join([str(pid),exe]),
                human_dst=':'.join([str(s) for s in dst_info[-2:]])
            )

    # Assume it's a file if there's something in the path section
    else: 
        if path := line['predicateObjectPath']: 
            path = path['string']
            g.add_edge(
                fmt_time(line['timestampNanos']),
                uuid,dst,
                path_to_tensor(exe,DEPTH),
                path_to_tensor(path,DEPTH),
                PROC,FILE, etype,
                human_src=':'.join([str(pid),exe]),
                human_dst=path
            )


def build_graph():
    files = glob.glob(HOME+'*-2.json*')
    files = [f for f in files if not f.endswith('.tar.gz')]
    
    with open('preprocessing/e3_ids.pkl', 'rb') as f:
        ids = pickle.load(f)

    g = FullGraph(0)

    # Can maybe parallelize this, but idk
    for i,file in enumerate(files):
        f = open(file, 'r')
        line = f.readline()

        prog = tqdm(desc='(%d/%d)' % (i+1,len(files)))
        while(line):
            line = json.loads(line)
            parse_line(line['datum'],g,ids)

            line = f.readline()
            prog.update()
        prog.close()

    g.finalize(max(EDGES.values()))
    with open('e3_graph.pkl', 'wb+') as f:
        pickle.dump(g, f)