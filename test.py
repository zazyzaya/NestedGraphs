import pickle 
from utils.perterbations import subgraph 

with open('/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/Sept23/benign/full_graph1.pkl', 'rb') as f:
    g = pickle.load(f)

subgraph(g)