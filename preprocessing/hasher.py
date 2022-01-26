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
    ], dim=0).unsqueeze(0)

def str_to_tensor(s: str) -> torch.Tensor:
    '''
    Converts a string to a tensor (dim always 8 for now.. unsure
    how to update this)


        Args: 
            s (str): string to be hashed
    '''
    b = abs(hash(s)).to_bytes(8, 'big')
    return torch.frombuffer(b, dtype=torch.int8) / 128


# # # # # # # # # # # # # # # # # # 
# Functions for the graph builder #
# # # # # # # # # # # # # # # # # #
 
def proc_feats(path: str, depth: int) -> torch.Tensor:
    '''
    Alias for path_to_tensor
    '''
    return path_to_tensor(path, depth)

def mod_feats(path: str, depth: int) -> torch.Tensor:
    '''
    Alias for path_to_tensor
    '''
    return path_to_tensor(path, depth)

file_acts = ['CREATE','DELETE','MODIFY','READ','RENAME','WRITE']
FILE_ACTIONS = {k:v for v,k in enumerate(file_acts)}
def file_feats(path: str, action: str, depth: int) -> torch.Tensor:
    '''
    Also includes action type in the encoding
    '''
    path = path_to_tensor(path, depth)
    action_t = torch.zeros((1, len(FILE_ACTIONS)))
    action_t[0][FILE_ACTIONS[action]]=1
    
    # Return path encoding and one-hot of how it was accessed
    return torch.cat([path,action_t], dim=1)


reg_acts = ['ADD', 'EDIT', 'REMOVE']
REG_ACTIONS = {k:v for v,k in enumerate(reg_acts)}
def reg_feats(path: str, action: str, depth: int) -> torch.Tensor:
    '''
    With registries, it seems like the deeper parts of the path contain more
    information than the early parts. So this is like path_to_tensor, but backwards.
    
    E.g. '\REGISTRY\MACHINE\SOFTWARE\...Schedule\TaskCache\Tasks\{732C6E77-1C2E-4875-A880-84ABC81F651D}\DynamicInfo'
    becomes  [
        hash(\REGISTRY\MACHINE\SOFTWARE\...Schedule\), 
        hash(TaskCache)
        hash(Tasks),
        hash({732...}),
        hash(DynamicInfo)
    ]
    '''
    levels = path.lower().rsplit('\\', depth)  # Trim off trailing \\
    levels = levels + ['']*(depth-len(levels))      # pad with empty entries if needed (NOTE: hash('') == 0)
    path = torch.cat([
        str_to_tensor(s) for s in levels
    ], dim=0).unsqueeze(0)

    action_t = torch.zeros((1, len(REG_ACTIONS)))
    action_t[0][REG_ACTIONS[action]]=1
    
    return torch.cat([path,action_t], dim=1)


if __name__ == '__main__': 
    path_to_tensor('\\Device\\HarddiskVolume1\\ProgramData\\lwa\\.winlogbeat.yml.new', 6)