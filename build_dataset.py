import glob 
import sys 
import pickle as pkl
from preprocessing.build_hostgraph import build_full_graphs
from preprocessing.build_unfiltered import parse_all

'''
DAY = 23#int(sys.argv[1])

# Mal host IDS for given day TODO last two days
mids = {
    23: [201,402,660,104,205,321,255,355,503,462,559,419,609,771,955,874,170]
}

# Malicious
build_full_graphs(mids[DAY], DAY, is_mal=True, jobs=17)

# Benign
bids = list(set(range(1,726)) - set(mids[DAY]))
build_full_graphs(bids, DAY)
'''
parse_all(8)
