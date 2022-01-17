import numpy as np
import torch 

def path_to_tensor(path: str, depth: int, dim: int, 
    delimeter: str='\\') -> torch.Tensor:
    '''
    Takes a file path  and creates a tensor composed of depth+1 hashes. 
    E.g. path_to_tensor('C:\\x\\y\\z\\abcd.txt') returns
    torch.cat([hash(C:), hash(x), hash(y), hash(z\\abcd.txt)]

        Args:
            path (str): A filepath to be hashed
            depth (int): The number of directories to individually hash
            dim (int): The dimension of individual hashes. Output will have dimension dim*(depth+1)
            delemiter (str): The path delimeter 
    '''
    pass 

def str_to_tensor(s: str, dim: int) -> torch.Tensor:
    '''
    Converts a string has to a tensor of dimesion dim

        Args: 
            s (str): string to be hashed
            dim (int): dimension of return tensor
    '''
    b = abs(hash(s)).to_bytes(4, 'little')
    print(b)
    