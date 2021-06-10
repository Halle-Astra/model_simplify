import json 
import numpy as np 
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
            t = t.replace(' ',',')
            t = t.split('\n')
            vertex_num, face_num, edge_num = np.array(t[1], dtype = np.float)
            
            if t[0].strip().lower() != 'off' or edge_num != 0:
                return [None,None]
            
            data = t[2:]
            v = data[:vertex_num]
            f = data[-face_num:]
            v = np.array(v,dtype = np.float)
            f = np.array(f,dtype = np.float)[:,1:]
        return v,f


conf = get_conf()
modelnet10 = ModelNet10(conf)
print(modelnet10.readfile('/data/sda/ModelNet10/night_stand/train/night_stand_0160.off'))
