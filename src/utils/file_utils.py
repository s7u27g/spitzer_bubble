import json
import shutil
import zipfile
import pickle
import numpy as np

def make_zipfile(directory):
    shutil.make_archive(directory, 'zip', root_dir=directory)
    return

def unzip_zipfile(file):
    with zipfile.ZipFile(file_name+'.zip') as zf:
        zf.extractall(file_name)
        pass
    return

def open_json(file):
    with open(file, 'r') as f:
        infos = json.load(f)
        pass
    return infos

def save_json(file, dictlist):
    with open(file, 'w') as f:
        json.dump(dictlist, f, cls = MyEncoder)
        pass
    return

def open_pickle(file):
    with open(file, 'rb') as f:
        infos = pickle.load(f)
        pass
    return infos

def save_pickle(file, obj):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)
        pass
    return

def json2csv(file):
    pass


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
#         elif isinstance(obj, np.array):
#             return obj.tolist()
        else:
            #return super(MyEncoder, self).default(obj)
            return None
