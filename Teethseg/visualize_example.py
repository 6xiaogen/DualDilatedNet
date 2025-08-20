import argparse
import random
import numpy as np
from pathlib import Path
import trimesh
import os
from utils.teeth_numbering import color_mesh
import sklearn.metrics as metrics
from lightning.pytorch import seed_everything
from torch.utils.data.dataset import Dataset

from dataset.mesh_dataset import Teeth3DSDataset
from dataset.preprocessing import *
from models.DDGCNet1_Seg import UseDDGCNet1, DDGCNet1
from sklearn.metrics import confusion_matrix
from datetime import datetime  # 导入 时间 模块
from torch.utils.data import DataLoader

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(SEED)

torch.set_float32_matmul_precision('medium')

random.seed(SEED)

seed_everything(SEED, workers=True)


def get_model():
    # return LitDilatedToothSegmentationNetwork()
    return DDGCNet1()


# 对obj数据做处理（归一化），再转为.pt的标准数据格式
def get_dataset(train_test_split=1) -> Dataset:
    test = Teeth3DSDataset("data/3dteethseg", processed_folder=f'processed',
                           verbose=True,
                           pre_transform=PreTransform(classes=17),
                           post_transform=None, in_memory=False,
                           force_process=False, is_train=False, train_test_split=train_test_split)

    return test


def infer(ckpt_path, train_test_split=1, data_idx=0, use_gpu=True):
    print(
        f"Running inference on data index {data_idx} using checkpoint {ckpt_path} with train_test_split {train_test_split}. Use GPU: {use_gpu}")

    test_dataset = get_dataset(train_test_split)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)
    teethnumber = len(test_loader)
    print(f"test data number :{teethnumber}")
    # 加载模型
    loaded_model = UseDDGCNet1.load_from_checkpoint(ckpt_path)
    if use_gpu:
        loaded_model = loaded_model.cuda()

    num_classes = 17
    loaded_model.eval()  # 测试阶段的数据 不会 参与训练，只是用于评估
    Auc_currentallnumber = 0  # 总的预测准确的点个数
    test_allmiou = []
    train_true_cls = []
    train_pred_cls = []
    idx = 0
    for pos, data, seg, orign_data in test_loader:
        real_label = seg.cpu().numpy()
        # 预测
        pre_labels = loaded_model.predict_labels((pos, data)).cpu().numpy()  # [1,30000]
        # 单个Auc准确的点数
        train_true_cls.append(real_label.reshape(-1))  # (batch_size * num_points)
        train_pred_cls.append(pre_labels.reshape(-1))  # (batch_size * num_points)

        # miou
        for shape_idx0 in range(real_label.shape[0]):
            part_ious = []
            for cls in range(num_classes):
                I = np.sum(np.logical_and(pre_labels[shape_idx0] == cls, real_label[shape_idx0] == cls))
                U = np.sum(np.logical_or(pre_labels[shape_idx0] == cls, real_label[shape_idx0] == cls))
                if U == 0:
                    iou = 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            test_allmiou.append(np.mean(part_ious))

        orign_data_pre = torch.zeros(pos.shape[1], 9).float()
        orign_data=orign_data.squeeze(0)
        orign_data_pre[:, :3] = orign_data[:, :3]
        orign_data_pre[:, 3:6] = orign_data[:, 3:6]
        orign_data_pre[:, 6:9] = orign_data[:, 6:9]
        pre_labels=pre_labels.squeeze(0)

        orign_data_pre = orign_data_pre.numpy()
        pre_labels = pre_labels.reshape(-1, 1)

        outputsdata = np.concatenate((orign_data_pre, pre_labels), axis=1)

        # 打开文件并写入数据
        folder_name = 'results'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # 打开文件并写入数据
        filename = f'{folder_name}/output_{idx}.txt'
        with open(filename, 'w') as f:
            for row in outputsdata:
                # 将每行数据转换为6位小数的字符串，并写入文件
                f.write(' '.join([f'{x:.6f}' for x in row]) + '\n')
        idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run visualization of an example from the dataset')
    parser.add_argument('--out_dir', type=str, help='Output directory where the mesh will be saved', default='predictions/1800d_160b')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--n_bit_precision', type=int, help='N-Bit precision', default=16)
    parser.add_argument('--train_test_split', type=int,help='Train test split option. Either 1 or 2', default=1)
    parser.add_argument('--data_idx', type=int, default=1)
    parser.add_argument('--save_mesh', type=bool, default=True)
    parser.add_argument('--ckpt', type=str,required=False,help='Checkpoint path', default='F:/01NUAA_work/Dilated_Dual_Net/1保存的模型/TeethSeg/chaepoch=1-step=8.ckpt')

    args = parser.parse_args()
    infer(args.ckpt, args.train_test_split, args.data_idx,use_gpu=args.use_gpu)



