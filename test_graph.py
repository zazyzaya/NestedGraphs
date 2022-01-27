import sys
import time
import pickle as pkl
from preprocessing.build_hostgraph import build_graph

if len(sys.argv) > 2: 
    host = sys.argv[2]
else:
    host = 201

print("Building..")
s = time.time() 
g,nl = build_graph(host)
print("Graph built in %0.2fs" % (time.time()-s))

with open('graph%d.pkl' % host, 'wb+') as f:
    pkl.dump(g,f, protocol=pkl.HIGHEST_PROTOCOL)
with open('nodes%d.pkl' % host, 'wb+') as f:
    pkl.dump(nl,f, protocol=pkl.HIGHEST_PROTOCOL)