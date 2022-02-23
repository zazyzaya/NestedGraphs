import argparse
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

def test_emb(nodes, graph, model, model_path=HOME+'saved_models/'):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    labels = propogate_labels(graph,nodes)
    
    emb = torch.load(model_path+'embedder/emb%s.pkl' % model)
    desc = torch.load(model_path+'embedder/desc%s.pkl' % model)

    with torch.no_grad():
        data = sample(nodes)
        zs = emb(data, graph)
        preds = desc(zs, graph)

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)
    labels = labels[idx]

    auc_score = auc(labels.clamp(0,1), vals)
    ap_score = ap(labels.clamp(0,1), vals)

    with open(HOME+"predictions/embedder/preds%d%s.csv" % (graph.gid, model), 'w+') as f:
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

def test_det(embs, nodes, graph, model, model_path=HOME+'saved_models/'):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    labels = propogate_labels(graph,nodes)
    
    disc = torch.load(model_path+'detector/disc%s.pkl' % model)

    with torch.no_grad():
        disc.eval()
        preds = disc(embs, graph)

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)
    labels = labels[idx]

    auc_score = auc(labels.clamp(0,1), vals)
    ap_score = ap(labels.clamp(0,1), vals)

    rev_auc = auc(labels.clamp(0,1), -vals)
    rev_ap = ap(labels.clamp(0,1), -vals)

    with open(HOME+"predictions/detector/preds%d%s.csv" % (graph.gid, model), 'w+') as f:
        aucap = "AUC: %f\tAP: %f\n" % (auc_score, ap_score)
        rev_aucap = "AUC': %f\tAP': %f\n" % (rev_auc, rev_ap)
        f.write(aucap + rev_aucap + '\n')

        for i in range(vals.size(0)):
            outstr = '%s\t%f\t%0.1f\n' % (
                inv_map[idx[i].item()],
                vals[i],
                labels[i]
            )

            f.write(outstr)
            print(outstr, end='')
        
        print()
        print(aucap+rev_aucap,end='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hostname', '-n',
        default=201, type=int
    )
    parser.add_argument(
        '--model', '-m',
        default=''
    )
    parser.add_argument(
        '--embedder',
        action='store_true'
    )
    args = parser.parse_args()

    print("Testing host %04d with %s model" % (args.hostname, args.model))
    with open(HOME+'inputs/mal/graph%d.pkl' % args.hostname, 'rb') as f:
        graph = pickle.load(f)
    with open(HOME+'inputs/mal/nodes%d.pkl' % args.hostname, 'rb') as f:
        nodes = pickle.load(f)
    with open(HOME+'inputs/mal/emb%d.pkl' % args.hostname, 'rb') as f:
        embs = pickle.load(f)

    if args.embedder:
        test_emb(nodes, graph, args.model)
    else:
        test_det(embs, nodes, graph, args.model)
