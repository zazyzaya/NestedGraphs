import pickle 
FNAME = '/home/isaiah/code/NestedGraphs/inputs/Sept23/mal/full_graph201.pkl'

with open(FNAME, 'rb') as f:
    g = pickle.load(f)

g.get_one_hop(19510)