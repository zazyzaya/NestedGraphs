import sys

import pickle 
import torch 
import datetime as dt
from zoneinfo import ZoneInfo
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from graph_utils import propogate_labels

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'

fmt_ts = lambda x : dt.datetime.fromtimestamp(x).astimezone(
    ZoneInfo('Etc/GMT+4')
).isoformat()[11:-6]

def sample(nodes):
    return {
        'regs': nodes.sample_feat('regs'),
        'files': nodes.sample_feat('files')
    }

def test_no_labels(nodes, graph, model_path=HOME+'saved_models/'):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    
    emb = torch.load(model_path+'emb.pkl')
    desc = torch.load(model_path+'desc.pkl')

    with torch.no_grad():
        data = sample(nodes)
        zs = emb(data)
        preds = desc(zs, graph.x, graph.edge_index)

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)

    with open(HOME+"predictions/preds%d.csv" % graph.gid, 'w+') as f:
        f.write('PID,anom_score\n')

        for i in range(vals.size(0)):
            outstr = '%s,%f,%s\n' % (
                inv_map[idx[i].item()],
                vals[i],
                fmt_ts(nodes[i].ts)
            )

            f.write(outstr)
            print(outstr, end='')

def test(nodes, graph, model, model_path=HOME+'saved_models/'):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    labels = propogate_labels(graph,nodes)
    
    emb = torch.load(model_path+'emb%s.pkl' % model)
    desc = torch.load(model_path+'desc%s.pkl' % model)

    with torch.no_grad():
        data = sample(nodes)
        zs = emb(data, graph)
        preds = desc(zs, graph)

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)
    labels = labels[idx]

    auc_score = auc(labels.clamp(0,1), vals)
    ap_score = ap(labels.clamp(0,1), vals)

    with open(HOME+"predictions/preds%d%s.csv" % (graph.gid, model), 'w+') as f:
        aucap = "AUC: %f\tAP: %f\n" % (auc_score, ap_score)
        f.write(aucap + '\n')

        for i in range(vals.size(0)):
            outstr = '%s\t%f\t%0.1f\n' % (
                inv_map[idx[i].item()],
                vals[i],
                labels[i]
            )

            f.write(outstr)
            print(outstr, end='')
        
        print()
        print(aucap,end='')

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        i = int(sys.argv[1])
    else:
        i = 201

    if len(sys.argv) >= 3:
        model = '_' + sys.argv[2]
    else:
        model = ''

    print("Testing host %04d with %s model" % (i, model))
    with open(HOME+'inputs/mal/graph%d.pkl' % i, 'rb') as f:
        graph = pickle.load(f)
    with open(HOME+'inputs/mal/nodes%d.pkl' % i, 'rb') as f:
        nodes = pickle.load(f)

    test(nodes, graph, model)
