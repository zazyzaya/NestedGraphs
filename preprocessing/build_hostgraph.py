from dateutil.parser import isoparse

import torch 
from torch_geometric.data import Data 

from parser import field_map, feat_map
from hasher import path_to_tensor

# Converts from ISO timestamp to UTC time since epoch
fmt_ts = lambda x : isoparse(x).timestamp()
        
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

    # Getters
    def get_files(self, time=0):
        return self.files.get_data(time)
    def get_regs(self, time=0):
        return self.regs.get_data(time)

    # Setters
    def add_file(self, ts, file):
        self.files.add_data(ts, file)
    def add_reg(self, ts, reg):
        self.regs.add_data(ts, reg)

    # Clean up in preparation for analysis
    def finalize(self):
        self.files.defrag()
        self.regs.defrag()


class HostGraph(Data):
    def __init__(self, x=None, y=None, pos=None, normal=None, face=None, **kwargs):
        super().__init__(x, None, None, y, pos, normal, face, **kwargs)

        # Will be filled and converted to tensors eventually
        self.src, self.dst = [],[] # To be converted to edge_index
        self.edge_attr = []
        self.x = []

        # Give procs unique IDs starting at 0
        self.node_map = dict() 
        self.num_nodes = 0
        self.nodes = []

    def add_node(self, ts, pid, feat):
        if pid in self.node_map:
            return 

        self.node_map[pid] = self.num_nodes 
        
        self.x.append(feat)
        self.nodes.append(Node(self.num_nodes, ts))

        self.num_nodes += 1

    def add_edge(self, ts, pid, ppid, feat):
        # This process will be a root, since it's parent has unobserved feats
        if ppid not in self.node_map:
            self.add_node(ts, pid, feat)
        else: 
            self.add_node(ts, pid, feat)
            self.src.append(self.node_map[ppid])
            self.dst.append(self.node_map[pid])
            self.edge_attr.append(ts)
    
    def add_file(self, ts, pid, file):
        self.nodes[self.node_map[pid]].add_file(ts, file)

    def add_reg(self, ts, pid, reg):
        self.nodes[self.node_map[pid]].add_reg(ts, reg)
            

def parse_line(data: HostGraph, line: str) -> None:
    '''
    Adds process node to the graph if needed, along with imgpath feature

        Args: 
            data (HostGraph): data object containing the graph
            line (str): line from hostlog csv file
    '''
    fields = line.split(',', 3)
    ts, obj, act = fields[:3]

    if obj == 'PROCESS':
        if act != 'CREATE':
            return 
        