import json 
from easydict import EasyDict 

def get_conf():
    with open('conf.json') as conf_fobj:
        conf = json.load(conf_fobj)
        conf = EasyDict(conf)
    return conf

class ModelNet10():
    def __init__(self,conf):
        self.root = conf.Datasets.ModelNet10
    
    def read_file(self,filename):
        with open(filename) as f :
            t = f.read()
            t = t.split()
            t = 


conf = get_conf()
modelnet10 = ModelNet10(conf)
