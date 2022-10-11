import pickle
import glob 

HOME = 'inputs/Sept23/mal/'
files = glob.glob(HOME+'*')

def check_file(fname):
    with open(fname, 'rb') as f:
        g = pickle.load(f)

    for i in range(g.x.size(0)):
        g.get(i)

for f in files:
    check_file(f)

#from preprocessing.build_hostgraph import build_full_graph
#build_full_graph(0,1,503,23,True,False)