from collections import defaultdict
import pickle

from joblib import Parallel, delayed
import torch 
from tqdm import tqdm 


DAY = 23
JOBS = 16
HOME = '/mnt/raid0_24TB/datasets/NCR2/nested_optc/hosts/'
HOME = HOME+'Sept%d/' % DAY 
DLLS = HOME+'../../../unique_modules.txt'

# Update later for full dataset
HOSTS = list(range(1,25))
IN_FILES = [HOME+'sysclient%04d_mods.csv' % i for i in HOSTS]
OUT_DIR = 'output/'

get_id = lambda x : int(x.split('/')[-1].split('_')[0].replace('sysclient',''))

def get_mod_map():
    with open(DLLS, 'r') as f:
        mods = f.read().split('\n')[:-1]

    mod_map = dict() 
    for i,m in enumerate(mods):
        mod_map[m] = i

    return mod_map 

# Record format: 
# [ts, actorID, dll]
def build_one(fname, num, tot):
    host = defaultdict(list)
    db = get_mod_map()

    f = open(fname, 'r')
    line = f.readline() 

    prog = tqdm(desc='(%d/%d)' % (num+1,tot))
    while(line):
        tokens = line.split(',')
        
        # Temporary fix. Parser accidentally added extra lines
        # to the module file
        if len(tokens) > 3:
            line = f.readline()
            continue 

        _,uuid,dll = tokens
        
        dll = db[dll.strip()]
        host[uuid].append(dll)

        prog.update() 
        line = f.readline()
    f.close() 

    host_tensors = dict() 
    for k,v in host.items():
        host_tensors[k] = torch.tensor(v)
    
    torch.save(host_tensors, OUT_DIR+'modlist_%d.pkl' % get_id(fname))

def build_all():
    Parallel(n_jobs=JOBS, prefer='processes')(
        delayed(build_one)(f,i,len(IN_FILES)) 
        for i,f in enumerate(IN_FILES)
    )

if __name__ == '__main__':
    print(len(get_mod_map()), 'unique DLLs detected')
    build_all()