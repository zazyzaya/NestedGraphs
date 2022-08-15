import argparse
import sys

import pickle 
import torch 
import datetime as dt
from zoneinfo import ZoneInfo
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.metrics import precision_score, recall_score
from torch.distributions import Normal

from graph_utils import propogate_labels

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'
MS = 50 # Surprisingly quite consistant, even with sampling
torch.set_num_threads(8)

fmt_ts = lambda x : dt.datetime.fromtimestamp(x).astimezone(
    ZoneInfo('Etc/GMT+4')
).isoformat()[11:-6]

def sample(nodes, max_samples=MS):
    return {
        'regs': nodes.sample_feat('regs', max_samples=max_samples),
        'files': nodes.sample_feat('files', max_samples=max_samples)
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

def test_emb(nodes, graph, model_str, dim, model_path=HOME+'saved_models/', verbose=True, max_samples=None):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    labels = propogate_labels(graph,nodes)
    
    max_samples = max_samples if max_samples else MS
    emb = torch.load(model_path+'embedder/emb%s_%d.pkl' % (model_str, dim))
    desc = torch.load(model_path+'embedder/disc%s_%d.pkl' % (model_str, dim))

    with torch.no_grad():
        emb.eval()
        desc.eval()

        data = sample(nodes, max_samples=max_samples)
        zs = emb(data, graph)
        preds = desc(zs, graph)

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)
    labels = labels[idx]

    auc_score = auc(labels.clamp(0,1), vals)
    ap_score = ap(labels.clamp(0,1), vals)

    if verbose:
        with open(HOME+"predictions/embedder/preds%d%s.csv" % (graph.gid, model_str), 'w+') as f:
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

    return auc_score, ap_score

def test_emb_input(zs, nodes, graph, disc):
    labels = propogate_labels(graph,nodes)
    
    with torch.no_grad():
        disc.eval()
        preds = torch.sigmoid(disc(zs, graph).nan_to_num())

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)
    labels = labels[idx].clamp(0,1)

    auc_score = auc(labels, vals)
    ap_score = ap(labels, vals)

    top_k = [50, 100, 150, 200, 250]
    pr = {}
    for k in top_k: 
        preds = torch.zeros(vals.size())
        preds[:k] =1.
        
        p = precision_score(labels, preds)
        r = recall_score(labels, preds)

        pr[k] = (p,r)

    return auc_score, ap_score, pr

def test_gen(nodes, graph, model_str, dim, model_path=HOME+'saved_models/', verbose=True, max_samples=None):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    labels = propogate_labels(graph,nodes)
    
    max_samples = max_samples if max_samples else MS
    gen = torch.load(model_path+'embedder/gen%s_%d.pkl' % (model_str, dim))
    desc = torch.load(model_path+'embedder/disc%s_%d.pkl' % (model_str, dim))

    with torch.no_grad():
        gen.eval()
        desc.eval()

        data = sample(nodes, max_samples=max_samples)
        zs = gen(graph)
        preds = desc(zs, graph)

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)
    labels = labels[idx]

    auc_score = auc(labels.clamp(0,1), vals)
    ap_score = ap(labels.clamp(0,1), vals)

    if verbose:
        with open(HOME+"predictions/embedder/preds%d%s.csv" % (graph.gid, model_str), 'w+') as f:
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

    return auc_score, ap_score

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

def test_variational_gen(embs, nodes, graph, model, dim, model_path=HOME+'saved_models/detector/'):
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
        unnorm_dist = (mean_dist/std).unsqueeze(1) @ mean_dist.unsqueeze(-1)
        mahalanobis = -0.5 * unnorm_dist.squeeze(-1)
        dividend = mahalanobis.exp().squeeze(-1)
            
        k = mean.size(1)
        divisor = (2*torch.pi ** k) ** (1/2) * std.prod(dim=1)
        
        preds = dividend / (divisor + 1e-6)

    vals, idx = torch.sort(mahalanobis.squeeze(-1), descending=True)
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

    print(mahalanobis)
    print(dividend)
    print(divisor)
    print(vals.max(), vals.min())


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
        type=int, default=1
    )
    args = parser.parse_args()

    print("Testing host %04d with %s model" % (args.hostname, args.model))
    with open(HOME+'inputs/mal/graph%d.pkl' % args.hostname, 'rb') as f:
        graph = pickle.load(f)
    with open(HOME+'inputs/mal/nodes%d.pkl' % args.hostname, 'rb') as f:
        nodes = pickle.load(f)
    
    if not args.embedder:
        with open(HOME+'inputs/mal/emb%d_%d.pkl' % (args.hostname, args.dim), 'rb') as f:
            embs = pickle.load(f)
    else:
        embs = None

    if args.embedder:
        if args.generator:
            test_gen(embs, nodes, graph, args.model, args.dim, model_path=HOME+'saved_models/embedder/')
        else:
            test_emb(nodes, graph, args.model, args.dim)
    elif args.generator:
        test_gen(embs, nodes, graph, args.model, args.dim)
    else:
        test_det(embs, nodes, graph, args.model, args.dim)

