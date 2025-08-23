import argparse
import random
import numpy as np
import torch

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data.dataset import Dataset

from models.DDGCNet2_Seg import UseDDGCNet2
from models.DDGCNet2_Seg import DDGCNet2
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
    return UseDDGCNet2()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Training')
    parser.add_argument('--epochs', type=int,help='How many epochs to train', default=1)
    parser.add_argument('--tb_save_dir', type=str,help='Tensorboard save directory', default='save_logs')
    parser.add_argument('--experiment_name', type=str,help='Experiment Name',default='training')
    parser.add_argument('--experiment_version', type=str,help='Experiment Version',default='v1')
    parser.add_argument('--train_batch_size', type=int,help='Train batch size', default=1)
    parser.add_argument('--devices', nargs='+', help='Devices to use', default=[0])
    parser.add_argument('--n_bit_precision', type=int,help='N-Bit precision', default=16)
    parser.add_argument('--train_test_split', type=int,help='Train test split option. Either 1 or 2', default=1)
    parser.add_argument('--ckpt', type=str,required=False,help='Checkpoint path to resume training', default=None)
    parser.add_argument('--npoint', type=int, default=16000, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')

    args = parser.parse_args()

    print(f'Run Experiment using args: {args}')

    model = get_model()

    # data root
    root = 'F:/01NUAA_work/Dilated_Dual_Net/gitRealease/DDGCNet2/data'

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='train', normal_channel=args.normal)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)

    print("The number of training data is: %d" % len(TRAIN_DATASET))
    print("The number of test data is: %d" % len(TEST_DATASET))

    train_dataloader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.train_batch_size,
                                                   shuffle=True, drop_last=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=4)

    if args.experiment_name is None:
        experiment_name = f'{args.model}_threedteethseg'
    else:
        experiment_name = args.experiment_name

    experiment_version = args.experiment_version

    logger = TensorBoardLogger(args.tb_save_dir, name=experiment_name, version=experiment_version)

    log_dir = logger.log_dir
    print(f"log_dir:{log_dir}")

    checkpoint_callback = ModelCheckpoint(dirpath=log_dir, save_top_k=1, monitor="val_acc", mode='max')

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='cuda', devices=[int(d) for d in args.devices],
                         enable_progress_bar=True, logger=logger, precision=args.n_bit_precision,
                         callbacks=[checkpoint_callback], deterministic=False)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.ckpt)

