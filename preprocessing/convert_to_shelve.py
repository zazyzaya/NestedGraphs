import shelve 
import pickle
from tqdm import tqdm 

with open('proc_ids.pkl', 'rb') as f:
    db = pickle.load(f)
sh = shelve.open('database/proc_ids', 'c')

for i,(k,v) in tqdm(enumerate(db.items())):
    sh[k] = v 

sh.close()