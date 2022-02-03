import sys
import time
import pickle as pkl
from preprocessing.build_hostgraph import build_graph


if len(sys.argv) >= 2:
    host = sys.argv[1]
else: 
    host = '201'

print("Building host %s" % host)
s = time.time() 
g,nl = build_graph(int(host))
print("Graph built in %0.2fs" % (time.time()-s))

with open('graph%s.pkl' % host, 'wb+') as f:
    pkl.dump(g,f, protocol=pkl.HIGHEST_PROTOCOL)
with open('nodes%s.pkl' % host, 'wb+') as f:
    pkl.dump(nl,f, protocol=pkl.HIGHEST_PROTOCOL)