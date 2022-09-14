from turtle import forward
import torch 
from torch import nn 
from torch.distributed import rpc 
from torch.nn.parallel import DistributedDataParallel as DDP 

from .dist_utils import _remote_method_async, _param_rrefs, _remote_method
from .tgat import TGAT

def fair_scheduler(g, n_workers, max_cost):
    '''
    There's probably a better way to do this, but for now
    just using greedy method. Costs is a list of tuples
    s.t. each tuple is (d,id) where d is the degree of node
    and id refers to the node id in the graph
    '''
    costs = [
        (min(g.csr_ptr[n+1]-g.csr_ptr[n], max_cost), n)
        for n in range(g.x.size(0))
    ]

    jobs = [[] for _ in range(n_workers)]
    labor_scheduled = [0] * n_workers

    while(costs):
        to_schedule = costs.index(max(costs, key=lambda x : x[0]))
        give_to = labor_scheduled.index(min(labor_scheduled))

        cost, nid = costs.pop(to_schedule)
        labor_scheduled[give_to] += cost
        jobs[give_to].append(nid)

    return [torch.tensor(j) for j in jobs]

class WorkerDDP(DDP):
    def train(self, mode=True):
        '''
        For some reason, this method is not exposed in DDP by default
        so we need to extend it
        '''
        self.module.train(mode=mode)

    def forward(self, g, **kwargs):
        '''
        RPC Doesn't allow sending GPU tensors from remote, so must copy them 
        to cpu first >> 
        '''
        zs = super().forward(g, **kwargs)
        return zs.cpu()

class Coordinator(nn.Module):
    def __init__(self, workers: list, neighborhood_size: int, device=torch.device('cpu')):
        '''
        Takes in a list of WorkerDDP models and coordinates their behavior
        Needs to know how many neighbors each TGAT samples during its
        forward pass for better job scheduling
        '''
        super().__init__()

        self.workers = workers 
        self.n_workers = len(workers) 
        self.neighborhood_size = neighborhood_size
        self.device=device 

    def forward(self, graph, start_t=0., end_t=float('inf')):
        batches = fair_scheduler(graph, self.n_workers, self.neighborhood_size)
        jobs = self.embed(graph, batches, start_t, end_t)

        zs = [j.wait() for j in jobs]
        reidx = torch.zeros((graph.x.size(0),zs[0].size(1)), device=self.device)

        # Need to put embeddings back in order
        for i,z in enumerate(zs): 
            reidx[batches[i]] = z 

        return reidx 


    def embed(self, graph, batches, start_t, end_t):
        futures = []
        for i in range(self.n_workers):
            futures.append(
                _remote_method_async(
                    WorkerDDP.forward, 
                    self.workers[i],
                    graph, 
                    batch=batches[i],
                    start_t=start_t, 
                    end_t=end_t
                )
            )

        return futures 

    def parameter_rrefs(self):
        '''
        Distributed optimizer needs RRefs to params rather than the literal
        locations of them that you'd get with self.parameters(). This returns
        a parameter list of all remote workers and an RRef of the RNN held by
        the recurrent layer
        '''
        params = []
        for rref in self.workers: 
            params.extend(
                _remote_method(
                    _param_rrefs, rref
                )
            )
        
        return params

    def state_dict(self):
        return _remote_method(DDP.state_dict, self.workers[0])


    def load_state_dict(self, state_dict):
        jobs = [
            _remote_method_async(
                DDP.load_state_dict, 
                rref, 
                state_dict
            )
            for rref in self.workers
        ]

        [j.wait() for j in jobs]


    def train(self, mode=True):
        [
            _remote_method(
                WorkerDDP.train, 
                rref,
                mode=mode 
            )
            for rref in self.workers
        ]

    def eval(self):
        [
            _remote_method(
                WorkerDDP.train, 
                rref, 
                mode=False
            )
            for rref in self.workers
        ]


def get_worker(*args, **kwargs):
    return WorkerDDP(
        TGAT(*args, **kwargs)
    )