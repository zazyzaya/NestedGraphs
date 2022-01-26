import time
import pickle as pkl
from preprocessing.build_hostgraph import build_graph

print("Building..")
s = time.time() 
g = build_graph(201)
print("Graph built in %0.2fs" % (time.time()-s))

with open('graph201.pkl', 'wb+') as f:
    pkl.dump(g,f, protocol=pkl.HIGHEST_PROTOCOL)