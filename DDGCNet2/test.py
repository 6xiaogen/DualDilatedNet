import argparse
import random
import numpy as np
import torch
from pathlib import Path
import trimesh
from utils.teeth_numbering import color_mesh
import sklearn.metrics as metrics
from lightning.pytorch import seed_everything
from torch.utils.data.dataset import Dataset
import open3d as o3d

from models.DDGCNet2_Seg import UseDDGCNet2, DDGCNet2
from sklearn.metrics import confusion_matrix
from datetime import datetime
from torch.utils.data import DataLoader
from data_utils.ShapeNetDataLoader import PartNormalDataset

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
    return DDGCNet2()


def infer2(ckpt_path, train_test_split=1, data_idx=0, save_mesh=False, out_dir='plots', use_gpu=True):
    print(
        f"Running inference on data index {data_idx} using checkpoint {ckpt_path} with train_test_split {train_test_split}. Use GPU: {use_gpu}")

    root = 'F:/01NUAA_work/Dilated_Dual_Net/Arch_SegGroup/data/'

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    print("The number of test data is: %d" % len(TEST_DATASET))
    test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0)

    teethnumber = len(test_loader)
    loaded_model = UseDDGCNet2.load_from_checkpoint(ckpt_path)
    if use_gpu:
        loaded_model = loaded_model.cuda()
    num_classes = 2
    loaded_model.eval()
    Auc_currentallnumber = 0
    test_allmiou = []
    train_true_cls = []
    train_pred_cls = []
    for pos, data, seg, orign_data, seg_group in test_loader:

        real_label = seg.cpu().numpy()

        pre_labels = loaded_model.predict_labels((pos, data, seg, orign_data, seg_group)).cpu().numpy()
        pre_labels = pre_labels.reshape(real_label.shape[0],-1)
        train_true_cls.append(real_label.reshape(-1))  # (batch_size * num_points)
        train_pred_cls.append(pre_labels.reshape(-1))  # (batch_size * num_points)
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

    # Auc
    train_true_cls = np.concatenate(train_true_cls)
    train_pred_cls = np.concatenate(train_pred_cls)
    train_AllAuc = metrics.accuracy_score(train_true_cls, train_pred_cls)

    # miou
    miou_allteeth = np.nanmean(test_allmiou)

    print(f"Total Accuracy (AUC): {train_AllAuc:.4f}")
    print(f"Mean mIoU for all teeth: {miou_allteeth:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run visualization of an example from the dataset')
    parser.add_argument('--out_dir', type=str,help='Output directory where the mesh will be saved', default='predictions/1800d_160b')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--n_bit_precision', type=int,help='N-Bit precision', default=32)
    parser.add_argument('--train_test_split', type=int,help='Train test split option. Either 1 or 2', default=1)
    parser.add_argument('--data_idx', type=int, default=1)
    parser.add_argument('--save_mesh', type=bool, default=True)
    parser.add_argument('--ckpt', type=str,required=False,help='Checkpoint path', default='root.ckpt')
    parser.add_argument('--npoint', type=int, default=22000, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=True,help='Whether to use normal information [default: False]')

    args = parser.parse_args()
    infer2(args.ckpt, args.train_test_split, args.data_idx, args.save_mesh, args.out_dir,
          use_gpu=args.use_gpu)



