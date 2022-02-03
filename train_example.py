import pickle
import sys 
import time 

import torch 
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam 

from models.hostlevel import NodeEmbedder
from models.gan import NodeGenerator, GATDescriminator

torch.set_num_threads(16)

if len(sys.argv) >= 2:
    host = sys.argv[1]
else: 
    host = '201'

with open('nodes%s.pkl' % host, 'rb') as f:
    nodes = pickle.load(f)

with open('graph%s.pkl' % host, 'rb') as f:
    graph = pickle.load(f)


criterion = BCEWithLogitsLoss()
EMBED_SIZE = 16
TRUE_VAL = 0.
FALSE_VAL = 1. # False should be 1 as in an anomaly score

#torch.autograd.set_detect_anomaly(True)
def train_loop(nodes, graph, epochs):
    emb = NodeEmbedder(
        nodes.file_dim, 
        nodes.mod_dim, 
        nodes.reg_dim,
        16, 8, EMBED_SIZE
    )

    gen = NodeGenerator(graph.x.size(1), 32, 64, EMBED_SIZE)
    desc = GATDescriminator(EMBED_SIZE+graph.x.size(1), 16)

    e_opt = Adam(emb.parameters(), lr=0.01)
    g_opt = Adam(gen.parameters(), lr=0.01)
    d_opt = Adam(desc.parameters(), lr=0.01)

    data = nodes.sample()
    for e in range(epochs):
        start = time.time() 

        # Positive samples & train embedder
        e_opt.zero_grad()
        d_opt.zero_grad()
        real = emb.forward(data['files'], data['mods'], data['regs'])
        preds = desc.forward(torch.cat([real, graph.x], dim=1), graph.edge_index)
        r_loss = criterion(preds, torch.full((graph.num_nodes,1), TRUE_VAL))
        r_loss.backward()
        e_opt.step() 

        # Negative samples
        g_opt.zero_grad()
        fake = gen.forward(graph.x).detach()
        preds = desc.forward(torch.cat([fake, graph.x], dim=1), graph.edge_index)
        f_loss = criterion(preds, torch.full((graph.num_nodes,1), FALSE_VAL))
        f_loss.backward() 
        d_opt.step() 

        # Train generator
        fake = gen(graph.x)
        preds = desc.forward(torch.cat([fake, graph.x], dim=1), graph.edge_index)
        g_loss = criterion(preds, torch.full((graph.num_nodes,1), TRUE_VAL))
        g_loss.backward() 
        g_opt.step()

        elapsed = time.time() - start
        print(
            "[%d] Emb: %0.4f, Gen: %0.4f, Disc: %0.4f (%0.2fs)" 
            % (e, r_loss, (r_loss+f_loss)/2, g_loss, elapsed)
        )

        if e % 50: 
            torch.save(emb, 'saved_models/emb.pkl')
            torch.save(desc, 'saved_models/desc.pkl')
            torch.save(gen, 'saved_models/gen.pkl')

    torch.save(emb, 'saved_models/emb.pkl')
    torch.save(desc, 'saved_models/desc.pkl')
    torch.save(gen, 'saved_models/gen.pkl')

if __name__ == '__main__':
    train_loop(nodes, graph, 250)