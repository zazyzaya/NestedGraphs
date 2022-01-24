from dateutil.parser import isoparse

from parser import field_map, feat_map
from hasher import proc_feats, file_feats, reg_feats, mod_feats 
from datastructures import HostGraph

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
        graph.add_edge(ts, pid, ppid, proc_feats(path, ))

    elif obj == 'FILE':
        pid, ppid, path = feats[:-1]
        graph.add_file(ts, pid, file_feats(path, act, ))