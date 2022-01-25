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
from joblib import Parallel, delayed

# globals
JOBS=16

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
file_paths = []

for f in eval_23sept:
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


def ip2host_map(file_paths):
    ip2host = {}
    for path in file_paths:
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
                                # else:
                                #     if row['hostname'] != ip2host[dest_ip]:
                                #         print("Duplicate")
                                #         print("dest_ip :", dest_ip)
                                #         print(row)

                    if properties['direction'] == 'outbound':
                        if 'src_ip' in properties:
                            src_ip = properties['src_ip']
                            if is_valid_ip(src_ip):
                                if src_ip not in ip2host:
                                    ip2host[src_ip] = row['hostname']


def parse_flow(file_paths, ip2host):
    print("Total Host: ", len(list(ip2host.keys())))
    file_name = '/mnt/raid0_24TB/isaiah/data/nested_optc/23sept_flow.csv'
    for path in file_paths:
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
                                feature_vector = [pid, ppid, image_path]
                                is_row_selected = True

                    if properties['direction'] == 'outbound':
                        if 'dest_ip' in properties:
                            dest_ip = properties['dest_ip']
                            if is_valid_ip(dest_ip) and dest_ip in ip2host:
                                src = row['hostname']
                                dest = ip2host[dest_ip]
                                is_row_selected = True

                    if is_row_selected == True:
                        with open(file_name, "a+") as fa:
                            host2host_flow = [row['timestamp'], src, dest, [pid, ppid, image_path]]
                            writer = csv.writer(fa)
                            writer.writerow(host2host_flow)

file_loc = '/mnt/raid0_24TB/isaiah/data/nested_optc/'

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

            if is_row_selected == True:
                with open(file_loc + file_name, "a+") as fa:
                    parsed_row = [row['timestamp'], row['object'], row['action'], feature_vector]
                    writer = csv.writer(fa)
                    writer.writerow(parsed_row)

if __name__ == '__main__':
    # Load in all paths in parallel
    # Parallel(n_jobs=JOBS, prefer='processes')(
    #     delayed(load_group)(fid, path, total_files) for fid,path in enumerate(file_paths)
    # )

    ip2host = ip2host_map(file_paths)
    with open(file_loc + 'ip2host.pkl', 'wb') as f:
        pkl.dump(ip2host, f)
    #
    # with open(file_loc + "ip2host.pkl", "rb") as f:
    #     ip2host = pkl.load(f)

    parse_flow(file_paths, ip2host)

