# ******************************************************************************
# adversary_parser.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 4/5/22   Paudel     Initial version,
# ******************************************************************************
from tqdm import tqdm
import json, csv, glob, os
from zipfile import ZipFile
import tarfile

HOME='/mnt/raid0_24TB/rpaudel42/data/adversary/atomic/'
ATOMIC_FILES = '/mnt/raid0_24TB/datasets/Security-Datasets/datasets/atomic/windows/*/host'
ATOMIC_JSON_FILES = '/mnt/raid0_24TB/rpaudel42/Security-Datasets/atomic'
COMPOUND_FILES = ['data/adversary/apt29_evals_day1_manual_2020-05-01225525.json', 'data/adversary/apt29_evals_day2_manual_2020-05-02035409.json']
def process_file_events(row):
    # for file
    # EventID 2 -> file creation time change by process
    # EventID 6 -> file load (driver loaded)
    # EventId 11 -> file create
    # EventId 15 -> file stream created
    # EventId 23 -> File delete (saved in archive)
    # EventID 26 -> file delete (deleted totally)
    if 'ProcessId' in row:
        object = 'file'
        action = None
        if row['EventID'] == 2:
            action = 'modify'
        elif row['EventID'] == 6:
            action = 'load'
        elif row['EventID'] == 11:
            action = 'create'
        elif row['EventID'] == 15:
            action = 'stream'
        elif row['EventID'] == 23:
            action = 'delete'
        elif row['EventID'] == 26:
            action = 'delete'

        file_name = row['Hostname'].split('.')[0].lower() + '.csv'
        pid = row['ProcessId']
        # ppid = row['ExecutionProcessID']
        if 'ExecutionProcessID' in row:
            ppid = row['ExecutionProcessID']
        else:
            ppid = None

        if 'TargetFilename' in row:
            file_path = row['TargetFilename']
        else:
            file_path = None
        if 'Image' in row:
            image_path = row['Image']
        else:
            image_path = None

        if 'new_path' in row:
            new_path = row['new_path']
        else:
            new_path = None

        if file_path is not None:
            is_row_selected = True

        feature_vector = [pid, ppid, file_path, image_path, new_path]
        return is_row_selected, file_name, object, action, feature_vector
    return False, None, None, None, None

def process_proc_events(row):
    # for process
    # EventId 1 -> process created
    # EventID 5 -> Process Terminated
    # EventId 10 -> Process Accessed
    # EventID 17 -> pipe created (for inter-process communication)
    # EventID 18 -> Pipe connected (for inter-process communicated)
    if 'ProcessId' in row:
        is_row_selected = True
        object = 'process'
        action = None
        if row['EventID'] == 1:
            action = 'create'
        elif row['EventID'] == 5:
            action = 'terminate'
        elif row['EventID'] == 10:
            action = 'access'
        elif row['EventID'] == 17:
            action = 'pipe'
        elif row['EventID'] == 18:
            action = 'pipe'

        file_name = row['Hostname'].split('.')[0].lower() + '.csv'

        pid = row['ProcessId']
        if 'ExecutionProcessID' in row:
            ppid = row['ExecutionProcessID']
        else:
            ppid = None
        if 'TargetImage' in row:
            image_path = row['TargetImage']
        else:
            image_path = None
        if 'SourceImage' in row:
            parent_image_path = row['SourceImage']
        else:
            parent_image_path = None
        feature_vector = [pid, ppid, image_path, parent_image_path]
        return is_row_selected, file_name, object, action, feature_vector
    return False, None, None, None, None

def process_reg_events(row):
    # # Registry events
    # # EventID 12 -> Registry object add or delete
    # # EventId 13 -> Registry value set
    # # EventId 14 -> Registry object renamed
    if 'ProcessId' in row:
        is_row_selected = True
        object = 'registry'
        action = None
        if row['EventID'] == 12:
            action = 'add'
        elif row['EventID'] == 13:
            action = 'update'
        elif row['EventID'] == 14:
            action = 'rename'

        file_name = row['Hostname'].split('.')[0].lower() + '.csv'
        pid = row['ProcessId']

        if 'ExecutionProcessID' in row:
            ppid = row['ExecutionProcessID']
        else:
            ppid = None
        if 'TargetObject' in row:
            key = row['TargetObject']
            value = row['TargetObject'].split('\\')[-1]
        else:
            key = None
            value = None

        if 'Image' in row:
            image_path = row['Image']
        else:
            image_path = None

        feature_vector = [pid, ppid, key, value, image_path]
        return is_row_selected, file_name, object, action, feature_vector
    return False, None, None, None, None

def parse_data_files(DATA_FILES,  type):
    for datafile in glob.glob(DATA_FILES + '/*/*.json'):
        print("Data File: %s " % (datafile))
        # from pandas.io import json
        # df = json.read_json(path_or_buf=datafile, lines=True)
        # print("Event Types: %s" % (df.EventType.unique()))
        i = 0
        with open(datafile) as f:
            technique_name = datafile.split('/')[-2]
            sub_technique_name = os.path.basename(datafile).split('_20')[0]
            for line in tqdm(f):
                row = json.loads(line)
                is_row_selected = False
                if 'Message' in row:
                    if row['Message'].startswith('File'):
                        is_row_selected, file_name, object, action, feature_vector = process_file_events(row)
                    elif row['Message'].startswith('Process'):
                        is_row_selected, file_name, object, action, feature_vector = process_proc_events(row)
                    elif row['Message'].startswith('Registry '):
                        is_row_selected, file_name, object, action, feature_vector = process_reg_events(row)

                if is_row_selected == True:
                    i += 1
                    is_anomaly = 1  # if is_anomalous_log(row) else 0
                    with open(HOME + file_name, "a+") as fa:
                        if 'EventTime' in row:
                            parsed_row = [row['EventTime'], object, action, feature_vector, is_anomaly, technique_name, sub_technique_name]
                        else:
                            parsed_row = [row['TimeCreated'], object, action, feature_vector, is_anomaly, technique_name, sub_technique_name]
                        writer = csv.writer(fa)
                        writer.writerow(parsed_row)

            print("Total events: %d" % (i))

def extract_atomic_files():
    print('Extracting all atomic files...')
    for file_name in glob.glob(ATOMIC_FILES+"/*"):
        zip_file_path = os.path.dirname(file_name)
        json_file_path = ATOMIC_JSON_FILES + '/' + zip_file_path.split('/')[-2] + '/' + os.path.basename(file_name).split('.')[0]
        try:
            with ZipFile(file_name, 'r') as zipObj:
                zipObj.extractall(os.path.dirname(json_file_path))
        except:
            try:
                with tarfile.open(file_name, 'r') as tarobj:
                    tarobj.extractall(os.path.dirname(json_file_path))
            except:
                print("\nCannot parse:  %s" %(file_name))
                pass

if __name__ == "__main__":
    # extract_atomic_files()
    parse_data_files(DATA_FILES = ATOMIC_JSON_FILES, type='atomic')
    # parse_data_files(DATA_FILES= COMPOUND_FILES, type='compound')