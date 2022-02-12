import pickle 
import torch 

from graph_utils import propogate_labels

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'

def test(nodes, graph, model_path=HOME+'saved_models/'):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    labels = propogate_labels(graph,nodes)
    
    emb = torch.load(model_path+'emb.pkl')
    desc = torch.load(model_path+'desc.pkl')

    with torch.no_grad():
        data = nodes.sample()
        zs = emb(data)
        preds = desc(zs, graph.x, graph.edge_index)

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)

    with open("preds%d.csv" % graph.gid, 'w+') as f:
        f.write('PID,anom_score\n')

        for i in range(vals.size(0)):
            outstr = '%s,%f,%0.1f\n' % (
                inv_map[idx[i].item()],
                vals[i],
                labels[i]
            )

            f.write(outstr)
            print(outstr, end='')

if __name__ == '__main__':
    with open('/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/mal/graph201.pkl', 'rb') as f:
        graph = pickle.load(f)
    with open('/mnt/raid0_24TB/isaiah/code/NestedGraphs/inputs/mal/nodes201.pkl', 'rb') as f:
        nodes = pickle.load(f)

    test(nodes, graph)
