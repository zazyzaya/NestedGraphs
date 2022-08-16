import json 
import torch 
HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'

def propagate_labels(g, day, label_f=HOME+'inputs/manual_labels.txt'):
    with open(label_f, 'r') as f:
        anoms = json.loads(f.read())

    labels = torch.zeros(g.num_nodes)
    if str(g.gid) not in anoms or str(day) not in anoms[str(g.gid)]:
        return labels 

    # Assume that anomalous processes produce other anomalous processes
    # Luckilly, the graph structure is a tree so we (shouldnt) have to 
    # check for loops and can just do BFS to assign labels 
    domain = set() 
    for v in anoms[str(g.gid)][str(day)].values():
        domain.add(int(v['nid']))

    for d in domain:
        labels[d] = 1

    while domain:
        mal = domain.pop()
        children = g.edge_index[1][g.edge_index[0] == mal]

        for c in children:
            # Troubling that there are loops in here..
            if labels[c] == 0:
                domain.add(c.item())
                labels[c] = labels[mal]+1

    #labels[labels != 0] = 1/labels[labels!=0]
    return labels
