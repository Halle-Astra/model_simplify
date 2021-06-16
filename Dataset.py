import json 
from matplotlib import pyplot as plt 
import glob 
import os 
import numpy as np 
from easydict import EasyDict 

def get_conf():
    with open('conf.json') as conf_fobj:
        conf = json.load(conf_fobj)
        conf = EasyDict(conf)
    return conf

class BaseDataset():
    def __init__(self,conf):
        self.f_nums = conf.InputSize.face

    def padding(self,face_xyz):
        face_padding = np.zeros([self.f_nums,face_xyz.shape[1]])
        face_padding[:face_xyz.shape[0]] = face_xyz
        flags = np.zeros([self.f_nums,1])
        flags[:face_xyz.shape[0]] = 1

        return face_padding, flags


class ModelNet10(BaseDataset):
    def __init__(self,conf,mode = 'train'):
        super().__init__()
        self.root = conf.Datasets.ModelNet10
        self.filelist = self.get_list(mode)
    
    def get_list(self,mode = 'train'):
        categories = os.listdir(self.root)
        categories.remove('README.txt')

        filelist = []
        for category in categories :
            category_root = os.path.join(self.root,category,mode)
            filelist += glob.glob(category_root+'/*.off')
        return filelist

    def dataset_statistic(self):
        train_list = self.get_list('train')
        test_list = self.get_list('test')
        allfile = train_list + test_list

        v_nums, f_nums = [],[]
        for filepath in allfile:
            v_num, f_num = [i.shape[0] for i in self.read_file(filepath)]
            v_nums.append(v_num)
            f_nums.append(f_num)
        v_maxnum = max(v_nums)
        f_maxnum = max(f_nums)
        print('max num of vertices is\t', v_maxnum) # 502603
        print('max num of faces is \t', f_maxnum) # 403575

        plt.figure(figsize = (15,8))
        plt.hist(v_nums,100)
        plt.hist(f_nums,100)
        plt.savefig('dataset_size_statistic.png',dpi = 300)
        plt.close()

    def read_file(self,filename):
        with open(filename) as f :
            t = f.read()
            t = t.split('\n')
            file_format = t[0]
            t = [i.split() for i in t if i]
            vertex_num, face_num, edge_num = np.array(t[1], dtype = np.int)
            
            if file_format.strip().lower() != 'off' or edge_num != 0:
                print('The format of file {} is not announced as OFF or edge num is not 0.'.format(filename))
                return [None,None]
            
            data = t[2:]
            v = data[:vertex_num]
            f = data[-face_num:]
            v = np.array(v,dtype = np.float)
            f_tmp = np.array(f,dtype = np.float)
            if np.unique(f_tmp[:,0]).size == 1 and f_tmp[0,0] == 3:
                f = f_tmp[:,1:]
            else:
                print('The size of face is not uniform or the faces are not triangles.')
                return [None,None]

            if conf.InputSize.face is not None :
                if f.shape[0] > conf.InputSize.face :
                    print('The num of faces are larger than limitation {}.'.format(conf.InputSize.face))
                    return [None,None]

        return v,f


conf = get_conf()
modelnet10 = ModelNet10(conf)
modelnet10.dataset_statistic()
