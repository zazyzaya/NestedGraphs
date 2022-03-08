import argparse
import sys

import pickle 
import torch 
import datetime as dt
from zoneinfo import ZoneInfo
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from torch.distributions import Normal

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

def test_emb(nodes, graph, model, dim, model_path=HOME+'saved_models/'):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    labels = propogate_labels(graph,nodes)
    
    emb = torch.load(model_path+'embedder/emb%s_%d.pkl' % (model, dim))
    desc = torch.load(model_path+'embedder/disc%s_%d.pkl' % (model, dim))

    with torch.no_grad():
        emb.eval()
        desc.eval()

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

def test_det(embs, nodes, graph, model, dim, model_path=HOME+'saved_models/'):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    labels = propogate_labels(graph,nodes)
    
    disc = torch.load(model_path+'detector/disc%s_%d.pkl' % (model,dim))

    with torch.no_grad():
        disc.eval()
        preds = disc(embs, graph)

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)
    labels = labels[idx]

    auc_score = auc(labels.clamp(0,1), vals)
    ap_score = ap(labels.clamp(0,1), vals)

    with open(HOME+"predictions/detector/preds%d%s.csv" % (graph.gid, model), 'w+') as f:
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

def test_gen(embs, nodes, graph, model, dim, model_path=HOME+'saved_models/detector/'):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    labels = propogate_labels(graph,nodes)
    
    gen = torch.load(model_path+'gen%s_%d.pkl' % (model,dim))

    with torch.no_grad():
        gen.eval()
        mean, std = gen.get_distros(graph)
        mean_dist = (embs-mean)

        # f(x; mu, std) = exp{ -1/2 (x-mu).T Cov^-1 (x-mu)}
        #                   ------------------------------
        #                       sqrt{ (2pi)^k ||Cov|| }
        #
        # Note that we use naiive covariance, so these are diagonal mats
        # of std vectors. Inverse is just Diag(1/std) and determinant is Trace(std)

        # Dot product of each sample with itself
        if std.dim() == 2:
            unnorm_dist = (mean_dist/std).unsqueeze(1) @ mean_dist.unsqueeze(-1)
            mahalanobis = -0.5 * unnorm_dist.squeeze(-1)
            dividend = mahalanobis.exp().squeeze(-1)

        # Using covariance matrix
        else:
            dividend = (-0.5 * (mean_dist.unsqueeze(1) @ torch.inverse(std) @ mean_dist.unsqueeze(-1))).exp()
            dividend = dividend.squeeze(-1)
            print(dividend.max(), dividend.min())
            
        k = mean.size(1)
        divisor = (2*torch.pi ** k) ** (1/2) * std.prod(dim=1)
        
        preds = dividend / (divisor + 1e-6)

    vals, idx = torch.sort(1-preds.squeeze(-1), descending=True)
    labels = labels[idx]
    
    print(divisor)
    print(vals.max(), vals.min())

    auc_score = auc(labels.clamp(0,1), vals)
    ap_score = ap(labels.clamp(0,1), vals)

    with open(HOME+"predictions/detector/preds%d%s.csv" % (graph.gid, model), 'w+') as f:
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
    parser.add_argument(
        '--generator', 
        action='store_true'
    )
    parser.add_argument(
        '--dim', '-d',
        type=int, default=64
    )
    args = parser.parse_args()

    print("Testing host %04d with %s model" % (args.hostname, args.model))
    with open(HOME+'inputs/mal/graph%d.pkl' % args.hostname, 'rb') as f:
        graph = pickle.load(f)
    with open(HOME+'inputs/mal/nodes%d.pkl' % args.hostname, 'rb') as f:
        nodes = pickle.load(f)
    with open(HOME+'inputs/mal/emb%d_%d.pkl' % (args.hostname, args.dim), 'rb') as f:
        embs = pickle.load(f)

    if args.embedder:
        if args.generator:
            test_gen(embs, nodes, graph, args.model, args.dim, model_path=HOME+'saved_models/embedder/')
        else:
            test_emb(nodes, graph, args.model, args.dim)
    elif args.generator:
        test_gen(embs, nodes, graph, args.model, args.dim)
    else:
        test_det(embs, nodes, graph, args.model, args.dim)

