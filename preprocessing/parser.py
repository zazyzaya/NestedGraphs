import json
from tqdm import tqdm
import glob
import re
import gzip
import pickle as pkl
import csv
import pickle
from joblib import Parallel, delayed
from functools import reduce
from dateutil.parser import isoparse

# globals
JOBS=16
DAY =23
DB_HOME = '/mnt/raid0_24TB/datasets/NCR2/nested_optc/proc_ids.pkl'
HOME='/mnt/raid0_24TB/isaiah/data/nested_optc/%d/' % DAY 
IP_MAP=HOME+'ipmap.pkl'

# mappings
field_map = {
    'ts' : 0,
    'object': 1, 
    'action': 2, 
    'features': 3
}

feat_map = {
    'THREAD': {
        'src_pid': 0, 
        'src_tid': 1,
        'tgt_pid': 2, 
        'tgt_tid': 3
    },
    'FILE': {
        'pid': 0,
        'ppid': 1, 
        'file_path': 2, 
        'image_path': 3, 
        'new_path': 4
    },
    'PROCESS': {
        'pid': 0,
        'ppid': 1, 
        'image_path': 2,
        'parent_image_path': 3 
    },
    'REGISTRY': {
        'pid': 0,
        'ppid': 1, 
        'key': 2, 
        'value': 3, 
        'image_path': 4
    },
    'MODULE': {
        'pid': 0,
        'ppid': 1, 
        'image_path': 2,
        'module_path': 3
    }
}

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
                pid = row['pid']
                ppid = row['ppid']

                if 'image_path' in row['properties']:
                    image_path = row['properties']['image_path']
                    objects[row['objectID']] = (pid,image_path)

                if 'parent_image_path' in row['properties']:
                    parent_image_path = row['properties']['parent_image_path']
                    objects[row['actorID']] = (ppid, parent_image_path) 

    return objects 

def build_uuid_db():
    dbs = Parallel(n_jobs=JOBS, prefer='processes')(
        delayed(get_uuids)(fid, path, total_files) for fid,path in enumerate(file_paths)
    )

    # Merge into single dict
    out_db = {k:v for d in dbs for k,v in d.items()}
    with open('proc_ids.pkl', 'wb+') as f:
        pkl.dump(out_db, f)

def load_group(fid, file_path, total):
    with open('proc_ids.pkl', 'rb') as f:
        objects = pkl.load(f)

    with gzip.open(file_path, 'rb') as f:
        for line in tqdm(f, desc='%d/%d' % (fid, total)):
            is_row_selected = False
            row = json.loads(line.decode().strip())


            if row['object'] == 'FILE':
                file_name = row['hostname'].split('.')[0].lower()+'.csv'
                pid = row['pid']
                ppid = row['ppid']
                
                if 'image_path' in row['properties']:
                    image_path = row['properties']['image_path']
                else:
                    image_path = None
                if 'file_path' in row['properties']:
                    file_path = row['properties']['file_path']
                else:
                    file_path = None
                if 'new_path' in row['properties']:
                    new_path = row['properties']['new_path']
                else:
                    new_path = None

                # Need to make sure we can correlate back to process
                # that acted on this file. Otherwise, can't do anything with it
                if image_path is not None:
                    is_row_selected = True
                
                # See if it's been cached in the database of process info 
                # if it wasn't explicitly listed
                # Only bother querying db if filepath is also available
                elif file_path is not None: 
                    aid = row['actorID']
                    proc_info = objects.get(aid,None)

                    if proc_info is not None: 
                        pid,image_path = proc_info 
                        is_row_selected = True 

                feature_vector = [pid, ppid, file_path, image_path, new_path]
            
            if row['object'] == 'PROCESS': 
                file_name = row['hostname'].split('.')[0].lower()+'.csv'

                pid = row['pid']
                ppid = row['ppid']

                # Try to get info from file before querying large db 
                if 'image_path' in row['properties']:
                    image_path = row['properties']['image_path']
                else:
                    pid,image_path = objects.get(row['objectID'], (None,None))
                
                if 'parent_image_path' in row['properties']:
                    parent_image_path = row['properties']['parent_image_path']
                else:
                    ppid,parent_image_path = objects.get(row['actorID'], (None,None))

                if image_path is not None:
                    is_row_selected = True 

                feature_vector = [pid, ppid, image_path, parent_image_path]
                
            if row['object'] == 'REGISTRY':
                file_name = row['hostname'].split('.')[0].lower()+'.csv'
                pid = row['pid']
                ppid = row['ppid']

                if 'key' in row['properties']:
                    key = row['properties']['key']
                else:
                    key = None
                if 'value' in row['properties']:
                    value = row['properties']['value']
                else:
                    value = None
                if 'image_path' in row['properties']:
                    image_path = row['properties']['image_path']
                else:
                    image_path = None

                if image_path is None: 
                    # See if it's been cached in the database of process info 
                    # if it wasn't explicitly listed
                    aid = row['actorID']
                    proc_info = objects.get(aid,None)

                    if proc_info is not None: 
                        pid,image_path = proc_info 
                        is_row_selected = True 

                feature_vector = [pid, ppid, key, value, image_path]

            if is_row_selected:
                with open(HOME + file_name, "a+") as fa:
                    parsed_row = [row['timestamp'], row['object'], row['action'], feature_vector]#, is_anomaly]
                    writer = csv.writer(fa)
                    writer.writerow(parsed_row)

if __name__ == '__main__':
    # Load in all paths in parallel
    #build_uuid_db()
    Parallel(n_jobs=JOBS, prefer='processes')(
        delayed(load_group)(fid, path, total_files) for fid,path in enumerate(file_paths)
    )
    #load_group(0, file_paths[0], total_files)
    

    '''
    if os.path.exists(IP_MAP):
        f = open(IP_MAP, 'rb')
        ip2host = pkl.load(f)
        f.close() 
    else:
        ip2host = ip2host_map(file_paths)

    parse_flow(file_paths, ip2host)
    reduce_flows()
    '''
