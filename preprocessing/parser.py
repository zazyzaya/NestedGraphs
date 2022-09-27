import json
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
IP_MAP=HOME+'ipmap.pkl'

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

def load_group(fid, file_path, total):
    with open('proc_ids.pkl', 'rb') as f:
        objects = pkl.load(f)

    with gzip.open(file_path, 'rb') as f:
        for line in tqdm(f, desc='%d/%d' % (fid, total)):
            is_row_selected = False
            row = json.loads(line.decode().strip())

            if row['object'] == 'FILE':
                file_name = row['hostname'].split('.')[0].lower()+'.csv'
                pid,image_path = objects.get(row['actorID'], (None,None))
                
                file_path = row['properties'].get('file_path')
                new_path = row['properties'].get('new_path')
                
                # See if it's been cached in the database of process info 
                # if it wasn't explicitly listed
                if file_path is not None and image_path is not None: 
                    is_row_selected = True 

                feature_vector = [pid, -1, file_path, image_path, new_path]
            
            if row['object'] == 'PROCESS': 
                file_name = row['hostname'].split('.')[0].lower()+'.csv'

                pid,image_path = objects.get(row['objectID'], (None,None))
                ppid,parent_image_path = objects.get(row['actorID'], (None,None))

                if image_path is not None:
                    is_row_selected = True 

                # Useless to know a process was accessed by some unknown process
                if row['action'] == 'OPEN' and ppid is None:
                    is_row_selected = False 

                feature_vector = [pid, ppid, image_path, parent_image_path]
                
            if row['object'] == 'REGISTRY':
                file_name = row['hostname'].split('.')[0].lower()+'.csv'
                pid,image_path = objects.get(row['actorID'], (None, None))

                key = row['properties'].get('key')
                value = row['properties'].get('value')

                if image_path is not None: 
                   is_row_selected = True 

                feature_vector = [pid, -1, key, value, image_path]

            if is_row_selected:
                with open(HOME + file_name, "a+") as fa:
                    parsed_row = [row['timestamp'], row['object'], row['action'], feature_vector]
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
