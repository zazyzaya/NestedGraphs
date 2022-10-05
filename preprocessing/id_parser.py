import json
import os 
from tqdm import tqdm
import glob
import re
import gzip
import pickle as pkl
import csv
from joblib import Parallel, delayed

# globals
JOBS=16
DAY =23
DB_HOME = '/mnt/raid0_24TB/datasets/NCR2/nested_optc/proc_ids.pkl'
HOME='/mnt/raid0_24TB/isaiah/data/nested_optc/%d/' % DAY 

# Grab files we want
file_paths = glob.glob("/mnt/raid0_24TB/datasets/NCR2/ecar/evaluation/%dSep*/*/*.json*" % DAY)

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


file_paths.sort(key=natural_keys)
file_paths = sorted(file_paths)
total_files = len(file_paths)
print("Total Files: ", total_files)

def get_uuids(fid, file_path, total):
    objects = dict()

    with gzip.open(file_path, 'rb') as f:
        for line in tqdm(f, desc='%d/%d' % (fid, total)):
            row = json.loads(line.decode().strip())
            
            if row['object'] == 'PROCESS': 
                oid = row['objectID']
                aid = row['actorID']

                pid = row['pid']
                ppid = row['ppid']

                if objects.get(oid) is None:
                    if 'image_path' in row['properties'] and pid != -1:
                        image_path = row['properties']['image_path']
                        objects[row['objectID']] = (pid,image_path)

                if objects.get(aid) is None:
                    if 'parent_image_path' in row['properties'] and ppid != -1:
                        parent_image_path = row['properties']['parent_image_path']
                        objects[row['actorID']] = (ppid,parent_image_path) 


            if row['object'] == 'THREAD':
                pid = row['pid']
                
                aid = row['actorID']
                oid = row['objectID']

                image_path = row['properties'].get('image_path')
            
                if objects.get(aid) is None:
                    if image_path and pid != -1:
                        objects[row['actorID']] = (pid, image_path)
 
                if objects.get(oid) is None: 
                    if image_path and pid != -1:
                        objects[row['objectID']] = (pid, image_path)


            if row['object'] == 'MODULE':                
                aid = row['actorID']
                pid = row['pid']
                if image_path := row['properties'].get('image_path'):
                    objects[aid] = (pid, image_path)


    return objects 

def build_uuid_db():
    dbs = Parallel(n_jobs=JOBS, prefer='processes')(
        delayed(get_uuids)(fid, path, total_files) for fid,path in enumerate(file_paths)
    )

    # Merge into single dict
    out_db = {k:v for d in dbs for k,v in d.items()}
    with open('proc_ids.pkl', 'wb+') as f:
        pkl.dump(out_db, f)


build_uuid_db()