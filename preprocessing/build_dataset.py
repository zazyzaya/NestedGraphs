import sys 
import pickle as pkl
from .build_hostgraph import build_full_graphs

DAY = int(sys.argv[1])

# Benign
bids = list(range(1,26))
benign = build_full_graphs(bids, DAY)
for i in range(len(bids)):
    g = benign[i]
    id = bids[i]

    with open('inputs/Sept%d/benign/full_graph%d.pkl' % (DAY,id), 'wb+') as f:
        pkl.dump(g, f)

# Mal
mids = [201,402,660,104,205,321,255,355,503,462,559,419,609,771,955,874,170]
malicious = build_full_graphs(mids, DAY)
for i in range(len(mids)):
    g = malicious[i]
    id = mids[i]

    with open('inputs/Sept%d/mal/full_graph%d.pkl' % (DAY,id), 'wb+') as f:
        pkl.dump(g, f)