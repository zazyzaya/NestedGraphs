from argparse import ArgumentError
import torch 
from torch.nn.utils.rnn import pack_sequence
from torch_geometric.data import Data 
        
class DynamicFeatures():
    '''
    Object holding dynamic features. Should never be accessed
    by the user. If it is, I have done something terribly wrong
    in writing the other classes--please let me know.
    '''
    def __init__(self):
        self.feats = [] 
        self.times = []

    def __len__(self):
        self.defrag()
        return self.feats[0].size(0)

    # Avoid stacking tensors every time
    def get_data(self, time=0):
        if len(self.feats) == 0:
            return torch.tensor([]), torch.tensor([])

        self.defrag()
        if not time:
            return self.times[0], self.feats[0]
        
        # If desired, only access feats observed up until `time`
        indices = (self.times[0] <= time).squeeze()
        return self.times[0][indices], self.feats[0][indices]

    # Just mainitain a list until accessed
    def add_data(self, time: float, feat: torch.Tensor) -> None:
        self.feats.append(feat)
        self.times.append(torch.tensor([time]))

    # Make sure all tensors are stacked
    def defrag(self):
        if len(self.feats) > 1:
            # If this is not the first defrag
            if self.feats[0].dim() == 2:
                self.feats = [torch.cat([self.feats[0], torch.stack(self.feats[1:])], dim=0)]
                self.times = [torch.cat([self.times[0], torch.stack(self.times[1:])], dim=0)]
            else: 
                self.feats = [torch.stack(self.feats)]
                self.times = [torch.stack(self.times)]
        
        if len(self.feats) == 1 and self.feats[0].dim() == 1:
            self.feats = [self.feats[0].unsqueeze(0)]
            self.times = [self.times[0].unsqueeze(0)]

class Node():
    '''
    Object holding node features
    '''
    def __init__(self, nid, first_occurence):
        self.nid = nid
        self.ts = first_occurence
    
        self.files = DynamicFeatures()
        self.regs = DynamicFeatures()
        self.mods = DynamicFeatures()

    def __len__(self):
        return len(self.files), len(self.regs), len(self.mods)

    # Getters
    def get_files(self, time=0):
        return self.files.get_data(time)
    def get_regs(self, time=0):
        return self.regs.get_data(time)
    def get_mods(self, time=0):
        return self.mods.get_data(time)
    
    def getter(self, fn, time=0):
        '''
        Gets arbitrary value specified by fn 
        '''
        if fn == 'files': 
            return self.files.get_data(time)
        elif fn == 'regs': 
            return self.regs.get_data(time)
        elif fn == 'mods':
            return self.mods.get_data(time)
        else:
            raise ArgumentError("fn must be in ['files', 'regs', 'mods']")

    # Setters
    def add_file(self, ts, file):
        self.files.add_data(ts, file)
    def add_reg(self, ts, reg):
        self.regs.add_data(ts, reg)
    def add_mod(self, ts, mod):
        self.mods.add_data(ts, mod)

    # Clean up in preparation for analysis
    def finalize(self):
        self.files.defrag()
        self.regs.defrag()
        self.mods.defrag()


class NodeList():
    '''
    Object holding dynamic nodes (no edges)
    '''
    def __init__(self):
        # Give procs unique IDs starting at 0
        self.node_map = dict() 
        self.num_nodes = 0
        self.nodes = []

        self.file_dim = None 
        self.reg_dim = None 
        self.mod_dim = None

    def __getitem__(self, idx):
        if type(idx) == int:
            return self.nodes[idx]
        else:
            return self.nodes[self.node_map[idx]]

    def add_node(self, ts, pid):
        if pid in self.node_map:
            return False

        self.node_map[pid] = self.num_nodes 
        self.nodes.append(Node(self.num_nodes, ts))
        self.num_nodes += 1
        return True

    # Add features to nodes
    def add_file(self, ts, pid, file):
        if self.file_dim is None:
            self.file_dim = file.size(-1)
        if pid in self.node_map:
            self.nodes[self.node_map[pid]].add_file(ts, file)

    def add_reg(self, ts, pid, reg):
        if self.reg_dim is None:
            self.reg_dim = reg.size(-1)
        if pid in self.node_map:
            self.nodes[self.node_map[pid]].add_reg(ts, reg)

    def add_mod(self, ts, pid, mod):
        if self.mod_dim is None:
            self.mod_dim = mod.size(-1)
        if pid in self.node_map:
            self.nodes[self.node_map[pid]].add_mod(ts, mod)

    def finalize(self):
        [n.finalize() for n in self.nodes]


    def sample(self, ts=None, batch=[]):
        if not len(batch):
            batch = list(range(self.num_nodes))
        
        return {
            'files': self.sample_feat('files', batch=batch, ts=ts),
            'regs': self.sample_feat('regs', batch=batch, ts=ts),
            'mods': self.sample_feat('mods', batch=batch, ts=ts)
        }

    def sample_feat(self, fn, batch=[], ts=None):
        times, feats = [],[]
        indices = []

        if not len(batch):
            batch = list(range(self.num_nodes))

        for b in batch:
            t,x = self.nodes[b].getter(fn, time=ts)
            
            # Avoid errors from empty tensors
            if x.size(0):
                times.append(t)
                feats.append(x)
                indices.append(b)

        x = pack_sequence(feats, enforce_sorted=False)
        t = pack_sequence(times, enforce_sorted=False)
        i = torch.tensor(indices)

        return t,x,i
        

class HostGraph(Data):
    def __init__(self, gid, **kwargs):
        super().__init__(**kwargs)
        '''
        Object holding only edges, and intransient node features
        '''
        # Unique identifier
        self.gid = gid 

        # Will be filled and converted to tensors eventually
        self.src, self.dst = [],[] # To be converted to edge_index
        self.edge_attr = []
        self.x = []

        # Turns graph write-only after everything is built
        self.ready = False

    def add_node(self, ts, pid, feat, nodelist):
        assert not self.ready, 'add_node undefined after self.finalize() has been called'
        if nodelist.add_node(ts, pid):
            self.x.append(feat)

    def add_edge(self, ts, pid, ppid, feat, pfeat, nodelist):
        assert not self.ready, 'add_edge undefined after self.finalize() has been called'
         
        self.add_node(ts, ppid, pfeat, nodelist)
        self.add_node(ts, pid, feat, nodelist)
        self.src.append(nodelist.node_map[ppid])
        self.dst.append(nodelist.node_map[pid])
        self.edge_attr.append(ts)

    
    def finalize(self):
        '''
        Convert all lists being constructed during processing
        into tensors to be used in later processing
        '''
        # Preserve idempotence
        if self.ready:
            return 

        self.ready = True 
        self.edge_index = torch.tensor([self.src, self.dst])
        self.x = torch.cat(self.x, dim=0)
        self.edge_attr = torch.tensor(self.edge_attr)
        