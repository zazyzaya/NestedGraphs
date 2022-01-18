import numpy as np
import torch 

def path_to_tensor(path: str, depth: int, delimeter: str='\\') -> torch.Tensor:
    '''
    Takes a file path and creates a tensor composed of `depth` hashes. 
    E.g. path_to_tensor('C:\\x\\y\\z\\abcd.txt', 3) returns
    torch.cat([hash(C:), hash(x), hash(y\\z\\abcd.txt)]

        Args:
            path (str): A filepath to be hashed
            depth (int): The number of directories to individually hash
            delemiter (str): The path delimeter 
    '''
    levels = path.lower().split(delimeter, depth)[1:]   # Trim off leading \\
    levels = levels + ['']*(depth-len(levels))        # pad with empty entries if needed (NOTE: hash('') == 0)
    return torch.cat([
        str_to_tensor(s) for s in levels
    ], dim=0).unsqueeze(-1)

def str_to_tensor(s: str) -> torch.Tensor:
    '''
    Converts a string has to a tensor of dimesion dim

        Args: 
            s (str): string to be hashed
    '''
    b = abs(hash(s)).to_bytes(8, 'big')
    return torch.frombuffer(b, dtype=torch.int8) / 128
    
if __name__ == '__main__': 
    path_to_tensor('\\Device\\HarddiskVolume1\\ProgramData\\lwa\\.winlogbeat.yml.new', 6)