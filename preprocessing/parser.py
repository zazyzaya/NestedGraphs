import json
from tqdm import tqdm
import glob
import re
import gzip
import shelve
import pickle
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
    #objects = shelve.open('database/proc_ids', 'r')
    print("Loading ID database")
    with open('proc_ids.pkl','rb') as f:
        objects = pickle.load(f)

    mods = set()
    with gzip.open(file_path, 'rb') as f:
        for line in tqdm(f, desc='%d/%d' % (fid, total)):
            row = json.loads(line.decode().strip())

            if row['object'] == 'FILE':
                file_path = row['properties'].get('file_path')
                file_name = row['hostname'].split('.')[0].lower()+'.csv'
                feature_vector = [file_path]
            
            elif row['object'] == 'PROCESS': 
                pid,ppid = row['pid'], row['ppid']
                props = row['properties']

                # Avoid duplicate lines (often proc-opens are recorded a lot of times in a row)
                if row['action']==line[2] and [pid,ppid] == line[3][:2]:
                    continue 

                ip = props.get('image_path')
                pip = props.get('parent_image_path')

                # Only query if we have to 
                if ip is None:
                    pid,ip = objects.get(row['objectID'],(None,None))
                if pip is None: 
                    ppid,pip = objects.get(row['actorID'],(None,None))

                # Useless to know a process was accessed by some unknown process
                if row['action'] == 'OPEN' and ppid is None or ppid==-1:
                    continue 

                if ip is not None:
                    file_name = row['hostname'].split('.')[0].lower()+'.csv'
                    feature_vector = [pid, ppid, ip, pip]
                else:
                    continue 
            
            # Regs and mods write to separate files as they are used for 
            # feature generation in the processes later on
            elif row['object'] == 'REGISTRY':
                key = row['properties'].get('key','')
                file_name = row['hostname'].split('.')[0].lower()+'_regs.csv'

                if key:
                    with open(HOME + file_name, "a+") as fa:
                        parsed_row = [row['timestamp'], row['action'], row['actorID'], key]
                        writer = csv.writer(fa)
                        writer.writerow(parsed_row)
                continue 

            elif row['object'] == 'MODULE':
                dll = row['properties']['module_path'].split('\\')[-3:]
                
                # Ignore non-default DLLs (non-inductive learning ahead for these,
                # need guarantee of no unexpected DLLs in the future)
                if dll[0].lower() not in ['windows','systemroot']:
                    continue 

                # Need urlmon.dll == URLMON.DLL, e.g. 
                dll = dll[-1].lower()
                
                # For some reason, .exe's are also in here. But those are captured by 
                # the process entity, so only have linked libs here. 
                if dll.endswith('.dll'):
                    file_name = row['hostname'].split('.')[0].lower()+'_mods.csv'
                    mods.add(dll)

                    with open(HOME + file_name, "a+") as fa:
                        # Not using objectID b.c. it sees \\Windows\\...\\*.dll as separate from \\WINDOWS\\...\\*.DLL
                        parsed_row = [row['timestamp'], row['actorID'], dll]
                        writer = csv.writer(fa)
                        writer.writerow(parsed_row)
                
                continue

            else:
                continue

            with open(HOME + file_name, "a+") as fa:
                parsed_row = [row['timestamp'], row['object'], row['action'], row['actorID'], row['objectID'], *feature_vector]
                writer = csv.writer(fa)
                writer.writerow(parsed_row)

    return mods 

if __name__ == '__main__':
    # Load in all paths in parallel
    all_mods = Parallel(n_jobs=JOBS, prefer='processes')(
        delayed(load_group)(fid, path, total_files) for fid,path in enumerate(file_paths)
    )

    # Easier to build w2v model when all modules are known
    mods = set().union(*all_mods)
    with open(HOME+'unique_modules.txt', 'w+') as f:
        for m in mods:
            f.write(m+'\n')
