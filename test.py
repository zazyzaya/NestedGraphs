import pickle 

import torch 

from models.hostlevel import NodeEmbedder
from models.gan import GATDescriminator
from dist_train import EMBED_SIZE

def test(nodes, graph, model_path='saved_models/'):
    emb = torch.load(model_path+'emb.pkl')
    desc = torch.load(model_path+'desc.pkl')

    with torch.no_grad():
        data = nodes.sample()
        zs = emb(data)
        preds = desc(zs, graph.x, graph.edge_index)

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)
    inv_map = {v:k for k,v in nodes.node_map.items()}

    with open("preds%d.csv" % graph.gid, 'w+') as f:
        f.write('PID,anom_score\n')

        for i in range(vals.size(0)):
            outstr = '%s,%f\n' % (
                inv_map[idx[i].item()],
                vals[i]
            )

            f.write(outstr)
            print(outstr, end='')

if __name__ == '__main__':
    with open('inputs/mal/graph201.pkl', 'rb') as f:
        graph = pickle.load(f)
    with open('inputs/mal/nodes201.pkl', 'rb') as f:
        nodes = pickle.load(f)

    test(nodes, graph)
