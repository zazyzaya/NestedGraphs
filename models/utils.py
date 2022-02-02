import torch 
from torch.nn.utils.rnn import PackedSequence

def repack(data, x):
    '''
    Takes manipulated data `data` and puts it back into 
    a packed sequence with metadata matching `x`
    '''
    return PackedSequence(
        data, 
        batch_sizes=x.batch_sizes, 
        sorted_indices=x.sorted_indices, 
        unsorted_indices=x.unsorted_indices
    )

def packed_fn(x, fn):
    '''
    Helper fn to manipulate packed sequences (torch, pls fix this)
    '''
    return PackedSequence(
        fn(x.data), 
        batch_sizes=x.batch_sizes, 
        sorted_indices=x.sorted_indices, 
        unsorted_indices=x.unsorted_indices
    )

def packed_cat(seq, dim=1):
    '''
    Concatonates packed sequences
    '''
    x = seq[0]
    catted = torch.cat([packed.data for packed in seq], dim=dim)
    return PackedSequence(
        catted, 
        batch_sizes=x.batch_sizes, 
        sorted_indices=x.sorted_indices, 
        unsorted_indices=x.unsorted_indices
    )