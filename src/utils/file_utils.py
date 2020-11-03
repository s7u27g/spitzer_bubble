import json

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

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        else:
            #return super(MyEncoder, self).default(obj)
            return None