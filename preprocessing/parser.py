import json
import pandas as pd
from tqdm import tqdm
import datetime
import time
import numpy as np
import glob
import re
import gzip
import pickle as pkl
import ipaddress
import csv
import os 
from joblib import Parallel, delayed
from functools import reduce
from dateutil.parser import isoparse

# globals
JOBS=16
HOME='/mnt/raid0_24TB/isaiah/data/nested_optc/25/'
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
eval_23sept = glob.glob("/mnt/raid0_24TB/datasets/NCR2/ecar/evaluation/23Sep*/*/*.json*")
eval_24sept = glob.glob("/mnt/raid0_24TB/datasets/NCR2/ecar/evaluation/24Sep*/*/*.json*")
eval_25sept = glob.glob("/mnt/raid0_24TB/datasets/NCR2/ecar/evaluation/25Sep*/*/*.json*")

file_paths = []
for f in eval_25sept:
    file_paths.append(f)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


file_paths.sort(key=natural_keys)
file_paths = sorted(file_paths)
total_files = len(file_paths)
print("Total Files: ", total_files)


def is_ipv4(string):
    try:
        ipaddress.IPv4Network(string)
        return True
    except ValueError:
        return False


def is_multicast(ip_addr):
    if int(ip_addr.split('.')[0]) >= 224:
        return True
    else:
        return False


def is_broadcast(ip_addr):
    if '.255' in ip_addr:
        return True
    else:
        return False


def is_localhost(ip_addr):
    if int(ip_addr.split('.')[0]) == 127:
        return True
    else:
        return False


def is_valid_ip(ip_addr):
    if is_ipv4(ip_addr):
        if is_multicast(ip_addr) == False and is_broadcast(ip_addr) == False and is_localhost(ip_addr) == False:
            return True
    return False


def is_anomalous_log(row):
    host = row['hostname'].split('.')[0].lower()
    pid = int(row['pid'])
    timestamp = row['timestamp']
    if host == 'sysclient0201' and (pid == 5452 or pid == 2952) and (
            timestamp >= '2019-09-23T11:22:29.00-04:00' and timestamp <= '2019-09-23T13:27:29.00-04:00'):
        return True
    # Another case for 201 that remove registry persistance at 15:30
    if host == 'sysclient0201' and (pid == 5452 or pid == 2952) and (
            timestamp >= '2019-09-23T15:29:00.00-04:00' and timestamp <= '2019-09-23T15:32:00.00-04:00'):
        return True

    if host == 'sysclient0402' and pid == 3168 and (
            timestamp >= '2019-09-23T13:25:00.00-04:00' and timestamp <= '2019-09-23T13:36:00.00-04:00'):
        return True

    if host == 'sysclient0660' and pid == 880 and (
            timestamp >= '2019-09-23T13:38:00.00-04:00' and timestamp <= '2019-09-23T14:07:00.00-04:00'):
        return True

    if host == 'dc1' and pid == 1852 and (
            timestamp >= '2019-09-23T14:04:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0104' and pid == 3160 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0205' and pid == 5012 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0321' and pid == 2980 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0255' and pid == 3472 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0355' and pid == 1884 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0503' and pid == 1472 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0462' and pid == 2536 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0559' and pid == 1400 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0419' and pid == 1700 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0609' and pid == 3460 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0771' and pid == 4244 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0955' and pid == 4760 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0874' and pid == 5224 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    if host == 'sysclient0170' and pid == 644 and (
            timestamp >= '2019-09-23T14:44:00.00-04:00' and timestamp <= '2019-09-23T15:30:00.00-04:00'):
        return True

    return False


fmt_ts = lambda x : isoparse(x).timestamp()

def ip2host_map(file_paths):
    def map_ip_single_file(path):
        '''
        Callable for the Parallel object to execute
        '''
        ip2host = {}
        with gzip.open(path, 'rb') as f:
            for line in f:
                row = json.loads(line.decode().strip())
                if row['object'] == 'FLOW':
                    properties = row['properties']

                    if properties['direction'] == 'inbound':
                        if 'dest_ip' in properties:
                            dest_ip = properties['dest_ip']
                            if is_valid_ip(dest_ip):
                                if dest_ip not in ip2host:
                                    ip2host[dest_ip] = row['hostname']

                    if properties['direction'] == 'outbound':
                        if 'src_ip' in properties:
                            src_ip = properties['src_ip']
                            if is_valid_ip(src_ip):
                                if src_ip not in ip2host:
                                    ip2host[src_ip] = row['hostname']

        return ip2host 

    def reduce_dicts(accumulator, element):
        '''
        Helper function to combine ip maps generated by multiprocessing
        '''
        for k,v in element.items():
            # Overwrite if needed; but IPs *should* be unique? 
            accumulator[k] = v 
        return accumulator

    ipmaps = Parallel(n_jobs=JOBS, prefer='processes')(
        delayed(map_ip_single_file)(path) for path in file_paths
    )   
    ip2host = reduce(reduce_dicts, ipmaps, {})

    # Save the mapping in case we need it later
    f = open('/mnt/raid0_24TB/isaiah/data/nested_optc/ipmap.pkl', 'wb+')
    pkl.dump(ip2host, f, protocol=pkl.HIGHEST_PROTOCOL)
    f.close() 

    return ip2host


def parse_flow(file_paths, ip2host):
    def parse_file(pid, path):
        file_name = HOME+'flows/23sept_flow_%d.csv' % pid

        with gzip.open(path, 'rb') as f:
            for line in tqdm(f):
                row = json.loads(line.decode().strip())
                if row['object'] == 'FLOW':
                    is_row_selected = False
                    properties = row['properties']
                    pid = row['pid']
                    ppid = row['ppid']
                    if 'image_path' in properties:
                        image_path = properties['image_path']
                    else:
                        image_path = None

                    if properties['direction'] == 'inbound':
                        if 'src_ip' in properties:
                            src_ip = properties['src_ip']
                            if is_valid_ip(src_ip) and src_ip in ip2host:
                                src = ip2host[src_ip]
                                dest = row['hostname']
                                is_row_selected = True

                    if properties['direction'] == 'outbound':
                        if 'dest_ip' in properties:
                            dest_ip = properties['dest_ip']
                            if is_valid_ip(dest_ip) and dest_ip in ip2host:
                                src = row['hostname']
                                dest = ip2host[dest_ip]
                                is_row_selected = True

                    if is_row_selected == True:
                        is_anomaly = is_anomalous_log(row)
                        with open(file_name, "a+") as fa:
                            host2host_flow = [row['timestamp'], src, dest, [pid, ppid, image_path], is_anomaly]
                            writer = csv.writer(fa)
                            writer.writerow(host2host_flow)

    print("Total Host: ", len(list(ip2host.keys())))
    Parallel(n_jobs=JOBS, prefer='processes')(
        delayed(parse_file)(pid, path) for pid,path in enumerate(file_paths)
    )


def reduce_flows():
    paths = glob.glob(HOME+'flows/*.csv')
    files = [open(p, 'r') for p in paths]
    out_f = open(HOME+'flows.csv', 'w+')
    
    strip_time = lambda l : fmt_ts(l.split(',',1)[0])

    # There's probably a more efficient way to do this.. but we
    # only need to do it once. 
    lines = [f.readline() for f in files]
    times = np.array([strip_time(l) for l in lines])
    finished = 0

    prog = tqdm(desc='Lines sorted')
    while(finished != len(files)):
        lowest = np.argmin(times)
        out_f.write(lines[lowest])

        nextline = files[lowest].readline() 
        if nextline:
            times[lowest] = strip_time(nextline)
            lines[lowest] = nextline 
        else:
            times[lowest] = float('inf')
            files[lowest].close()
            finished += 1

        prog.update()

    out_f.close()


# Make this a callable so we can parallelize
def load_group(fid, file_path, total):
    with gzip.open(file_path, 'rb') as f:
        for line in tqdm(f, desc='%d/%d' % (fid, total)):
            is_row_selected = False
            row = json.loads(line.decode().strip())
            if row['object'] == 'THREAD':
                is_row_selected = True
                file_name = row['hostname'].split('.')[0].lower()+'.csv'
                src_pid = row['properties']['src_pid']
                src_tid = row['properties']['src_tid']
                tgt_pid = row['properties']['tgt_pid']
                tgt_tid = row['properties']['tgt_tid']
                feature_vector = [src_pid, src_tid, tgt_pid, tgt_tid]#, start_function, start_module_name, start_module]
   
            if row['object'] == 'FILE':
                file_name = row['hostname'].split('.')[0].lower()+'.csv'
                pid = row['pid']
                ppid = row['ppid']
                if 'file_path' in row['properties']:
                    file_path = row['properties']['file_path']
                else:
                    file_path = None
                if 'image_path' in row['properties']:
                    image_path = row['properties']['image_path']
                else:
                    image_path = None
                if 'new_path' in row['properties']:
                    new_path = row['properties']['new_path']
                else:
                    new_path = None
                if file_path is not None:
                    is_row_selected = True
                feature_vector = [pid, ppid, file_path, image_path, new_path]
            
            if row['object'] == 'PROCESS': 
                is_row_selected = True
                file_name = row['hostname'].split('.')[0].lower()+'.csv'
                pid = row['pid']
                ppid = row['ppid']
                if 'image_path' in row['properties']:
                    image_path = row['properties']['image_path']
                else:
                    image_path = None
                if 'parent_image_path' in row['properties']:
                    parent_image_path = row['properties']['parent_image_path']
                else:
                    parent_image_path = None
                feature_vector = [pid, ppid, image_path, parent_image_path]
                
            if row['object'] == 'REGISTRY':
                is_row_selected = True
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
                feature_vector = [pid, ppid, key, value, image_path]

            ''' No longer looking at modules
            if row['object'] == 'MODULE':
                is_row_selected = True
                file_name = row['hostname'].split('.')[0].lower()+'.csv'
                pid = row['pid']
                ppid = row['ppid']
                if 'image_path' in row['properties']:
                    image_path = row['properties']['image_path']
                else:
                    image_path = None
                if 'module_path' in row['properties']:
                    module_path = row['properties']['module_path']
                else:
                    module_path = None
                feature_vector = [pid, ppid, image_path, module_path]
            '''

            '''
            if is_row_selected == True:
                is_anomaly = 1 if is_anomalous_log(row) else 0
                with open(HOME + file_name, "a+") as fa:
                    parsed_row = [row['timestamp'], row['object'], row['action'], feature_vector]#, is_anomaly]
                    writer = csv.writer(fa)
                    writer.writerow(parsed_row)
            '''

if __name__ == '__main__':
    # Load in all paths in parallel
    Parallel(n_jobs=JOBS, prefer='processes')(
        delayed(load_group)(fid, path, total_files) for fid,path in enumerate(file_paths)
    )

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
