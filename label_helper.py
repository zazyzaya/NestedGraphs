import json 
import pickle
from tqdm import tqdm  
import datetime as dt 

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'
MAL_GRAPHS = [201,402,660,104,205,321,255,355,503,462,559,419,609,771,955,874,170]

# Schema:
#   Host:{ day: [pids] }
MAL_PROCS = {
    201: {23:[5452,2952]},
    402: {23:[3168]},
    660: {23:[880]},
    104: {23:[3160]},
    205: {23:[5012]},
    321: {23:[2980]},
    255: {23:[3472]},
    355: {23:[1884]},
    503: {23:[1472]},
    462: {23:[2536]},
    559: {23:[1400]},
    419: {23:[1700]},
    609: {23:[3460]},
    771: {23:[4244]},
    955: {23:[4760]},
    874: {23:[5224]},
    #170: {23:[604]} Not in the dataset(?)
}

maybe_mal = {}
for host in tqdm(MAL_PROCS.keys()): 
    maybe_mal[host] = dict()

    for day in MAL_PROCS[host].keys(): 
        maybe_mal[host][day] = dict() 
        mal = MAL_PROCS[host][day]

        with open(HOME+'inputs/Sept%d/mal/full_graph%d.pkl' % (day,host), 'rb') as f:
            graph = pickle.load(f)
        
        for pid,nid in graph.node_map.items():
            if graph.ntypes[nid] != graph.NODE_TYPES['PROCESS']:
                continue 

            pid,exe = pid.split(':')
            pid = int(pid)
            
            if pid in mal:
                print(pid,exe)
                maybe_pid = maybe_mal[host][day].get(pid, [])

                ts = dt.datetime.fromtimestamp(graph.node_times[nid].item()).isoformat()
                maybe_pid.append({'exe':exe,'ts':ts,'nid':nid})
                maybe_mal[host][day][pid] = maybe_pid

with open('inputs/maybe_mal.txt','w+') as f:
    f.write(json.dumps(maybe_mal, indent=2))