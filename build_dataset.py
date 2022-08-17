import sys 
import pickle as pkl
from preprocessing.build_hostgraph import build_graphs

DAY = int(sys.argv[1])

# Benign
bids = list(range(1,26))
benign = build_graphs(bids, DAY)
for i in range(len(bids)):
    g,n = benign[i]
    id = bids[i]

    with open('inputs/Sept%d/benign/graph%d.pkl' % (DAY,id), 'wb+') as f:
        pkl.dump(g, f)
    with open('inputs/Sept%d/benign/nodes%d.pkl' % (DAY,id), 'wb+') as f:
        pkl.dump(n, f)

'''
# Mal
mids = [201,402,660,104,205,321,255,355,503,462,559,419,609,771,955,874,170]
malicious = build_graphs(mids, DAY)
for i in range(len(mids)):
    g,n = malicious[i]
    id = mids[i]

    with open('inputs/Sept%d/mal/graph%d.pkl' % (DAY,id), 'wb+') as f:
        pkl.dump(g, f)
    with open('inputs/Sept%d/mal/nodes%d.pkl' % (DAY,id), 'wb+') as f:
        pkl.dump(n, f)
'''