# *_*coding:utf-8 *_*
import os # 对文件或者文件夹执行操作的模块
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc,m,centroid

def normal_normalize(normals):
    # 对法向量进行归一化处理
    norms = np.linalg.norm(normals, axis=1, keepdims=True)  # 计算每个法向量的模长
    normals = normals / (norms + 1e-8)  # 防止除零错误，进行归一化
    return normals

# 数据处理功能，可以对训练，测试，验证数据进行处理
class PartNormalDataset(Dataset):
    def __init__(self,root = './data', npoints=45000, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        # 开一个名为 synsetoffset2category.txt 的文件，该文件包含了类别名称的映射关系。
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        # with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
        #     train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        # with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
        #     val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        # with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
        #     test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        '''
        注释代码实现将测试数据文件名保存为txt
        '''
        # with open('./test_ids.txt',"w",encoding='utf-8') as f:
        #     for everyone in test_ids:
        #         f.write('\n'+everyone)
        #     f.close()
        # i = 0  # 从teeth中将模型的数据分类出来，自己去找到训练、测试、验证的模型，对于test、val文件没什么用
        for item in self.cat:    # 'Teeth'
            #print('category', item)
            self.meta[item] = []
            # dir_point = os.path.join(self.root, self.cat[item])   # data/Teeth  # 构建类别对应的文件夹路径，self.root 是数据集的根目录，self.cat[item] 是类别名称或标识符
            dir_point = self.root
            if split == 'train':
                dir_point = os.path.join(self.root, 'train_data')
            elif split == 'test':
                dir_point = os.path.join(self.root, 'test_data')
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)
            print(f"dir_point: {dir_point}")

            # 获取目录中的所有文件，并按存储顺序排列
            fns = sorted(os.listdir(dir_point))  # 默认按文件系统顺序返回
            # fns = sorted(os.listdir(dir_point), key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 按照文件名中的数字部分排序
            print(fns[0][0:-4])
            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token+'.txt'))
                # self.meta[item].append(os.path.join(dir_point, token ))
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))  #  [点云类别，点云对应类别的单个模型路径+名称]

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'Teeth': [0,1]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg, seg_group = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            seg_group = data[:, 6].astype(np.int32)

            # 删除牙齿分割标签1和16，即智齿
            valid_mask = (data[:, 6] != 1) & (data[:, 6] != 16)
            point_set = point_set[valid_mask]
            seg = seg[valid_mask]
            seg_group = seg_group[valid_mask]

        # 降采样
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]
        seg_group2 = seg_group[choice]   # 牙齿的展示分割牙号1-16, nd{30000}
        original_data = np.copy(point_set[:, 0:6])  # 备份原始点云数据信息

        # 对点坐标进行归一化处理
        point_set[:, 0:3], _, _ = pc_normalize(point_set[:, 0:3])  # # 对点坐标进行归一化处理
        point_set[:, 3:6] = normal_normalize(point_set[:, 3:6])  # 对法向量进行归一化
        pos = point_set[:, :3]

        # print(f"point_set shape:{point_set.shape}")
        return pos, point_set, seg, original_data, seg_group2

    def __len__(self):
        return len(self.datapath)



