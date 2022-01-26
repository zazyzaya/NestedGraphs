import torch 
from torch_geometric.data import Data 
        
class DynamicFeatures():
    def __init__(self):
        self.feats = [] 
        self.times = []

    # Avoid stacking tensors every time
    def get_data(self, time=0):
        self.defrag()

        if not time:
            return self.feats[0], self.times[0]
        
        # If desired, only access feats observed up until `time`
        indices = self.times[0] <= time 
        return self.times[0][indices], self.feats[0][indices]

    # Just mainitain a list until accessed
    def add_data(self, time: float, feat: torch.Tensor) -> None:
        self.feats.append(feat)
        self.times.append(torch.tensor([[time]]))

    # Make sure all tensors are stacked
    def defrag(self):
        if len(self.feats) > 1:
            self.feats = [torch.cat(self.feats, dim=0)]
            self.times = [torch.cat(self.times, dim=0)]


class Node():
    def __init__(self, nid, first_occurence):
        self.nid = nid
        self.ts = first_occurence
    
        self.files = DynamicFeatures()
        self.regs = DynamicFeatures()
        self.mods = DynamicFeatures()

    # Getters
    def get_files(self, time=0):
        return self.files.get_data(time)
    def get_regs(self, time=0):
        return self.regs.get_data(time)
    def get_mods(self, time=0):
        return self.mods.get_data(time)

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


class HostGraph(Data):
    def __init__(self, gid, x=None, y=None, pos=None, normal=None, face=None, **kwargs):
        super().__init__(x, None, None, y, pos, normal, face, **kwargs)
        
        # Unique identifier
        self.gid = gid 

        # Will be filled and converted to tensors eventually
        self.src, self.dst = [],[] # To be converted to edge_index
        self.edge_attr = []
        self.x = []

        # Give procs unique IDs starting at 0
        self.node_map = dict() 
        self.num_nodes = 0
        self.nodes = []

        # Turns graph write-only after everything is built
        self.ready = False

    def add_node(self, ts, pid, feat):
        assert not self.ready, 'add_node undefined after self.finalize() has been called'
        if pid in self.node_map:
            return 

        self.node_map[pid] = self.num_nodes 
        
        self.x.append(feat)
        self.nodes.append(Node(self.num_nodes, ts))

        self.num_nodes += 1

    def add_edge(self, ts, pid, ppid, feat):
        assert not self.ready, 'add_edge undefined after self.finalize() has been called'
        # This process will be a root, since it's parent has unobserved feats
        if ppid not in self.node_map:
            self.add_node(ts, pid, feat)
        else: 
            self.add_node(ts, pid, feat)
            self.src.append(self.node_map[ppid])
            self.dst.append(self.node_map[pid])
            self.edge_attr.append(ts)
    
    # Add features to nodes
    def add_file(self, ts, pid, file):
        self.nodes[self.node_map[pid]].add_file(ts, file)
    def add_reg(self, ts, pid, reg):
        self.nodes[self.node_map[pid]].add_reg(ts, reg)
    def add_mod(self, ts, pid, mod):
        self.nodes[self.node_map[pid]].add_mod(ts, mod)

    
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
        