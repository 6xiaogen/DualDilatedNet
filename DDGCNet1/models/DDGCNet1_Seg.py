from torch import nn
import torch
import torchmetrics as tm
from models.layer import BasicPointLayer, DualEdgeGraphConvBlock, DualDilatedEdgeGraphConvBlock, ResidualBasicPointLayer, \
    PointFeatureImportance, STNkd
import lightning as L


class DDGCNet1(nn.Module):
    def __init__(self, num_classes=17, feature_dim=15):
        super(DDGCNet1, self).__init__()
        self.num_classes = num_classes

        self.stnkd = STNkd(k=15)

        self.Dual_Dilated_block1 = DualEdgeGraphConvBlock(in_channels=feature_dim, hidden_channels=24,out_channels=24,
                                                             k=32, k_pos=0, k_dilated=12, k_dilated_pos=0,
                                                             edge_function=249)
        self.Far_Dual_Dilated_block1 = DualDilatedEdgeGraphConvBlock(in_channels=feature_dim, hidden_channels=36,
                                                                            out_channels=36,
                                                                            all_k=240, k=32, k_pos=0, k_dilated=12,
                                                                            k_dilated_pos=0,
                                                                            edge_function=249)

        self.Dual_Dilated_block2 = DualEdgeGraphConvBlock(in_channels=60, hidden_channels=48, out_channels=48,
                                                             k=32, k_pos=16, k_dilated=12, k_dilated_pos=6,
                                                             edge_function=249)
        self.Far_Dual_Dilated_block2 = DualDilatedEdgeGraphConvBlock(in_channels=60, hidden_channels=72,
                                                                            out_channels=72,
                                                                            all_k=360, k=32, k_pos=14, k_dilated=10,
                                                                            k_dilated_pos=6,
                                                                            edge_function=249)

        self.Dual_Dilated_block3 = DualEdgeGraphConvBlock(in_channels=120, hidden_channels=80, out_channels=80,
                                                             k=32, k_pos=16, k_dilated=12, k_dilated_pos=6,
                                                             edge_function=249)
        self.Far_Dual_Dilated_block3 = DualDilatedEdgeGraphConvBlock(in_channels=120, hidden_channels=120,
                                                                            out_channels=120,
                                                                            all_k=540, k=32, k_pos=14, k_dilated=10,
                                                                            k_dilated_pos=6,
                                                                            edge_function=249)

        self.mlp1 = BasicPointLayer(in_channels=380, out_channels=512)
        self.mlp2 = BasicPointLayer(in_channels=512, out_channels=1024)

        self.feature_importance = PointFeatureImportance(in_channels=1024)

        self.res_block1 = ResidualBasicPointLayer(in_channels=1024, out_channels=512, hidden_channels=512)
        self.res_block2 = ResidualBasicPointLayer(in_channels=512, out_channels=256, hidden_channels=256)

        self.out = BasicPointLayer(in_channels=256, out_channels=num_classes, is_out=1)

    def forward(self, x, pos):
        cd = torch.cdist(pos, pos)
        x = self.stnkd(x)

        x1_1, _ = self.Dual_Dilated_block1(x, pos)
        x1_2, _ = self.Far_Dual_Dilated_block1(x, pos, cd=cd)
        x1 = torch.cat([x1_1, x1_2], dim=2)

        x2_1, _ = self.Dual_Dilated_block2(x1, pos)
        x2_2, _ = self.Far_Dual_Dilated_block2(x1, pos, cd=cd)
        x2 = torch.cat([x2_1, x2_2], dim=2)

        x3_1, _ = self.Dual_Dilated_block3(x2, pos)
        x3_2, _ = self.Far_Dual_Dilated_block3(x2, pos, cd=cd)
        x3 = torch.cat([x3_1, x3_2], dim=2)

        x = torch.cat([x1, x2, x3], dim=2)  # n*240

        x = self.mlp1(x)
        x = self.mlp2(x)

        x = self.feature_importance(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.out(x)
        return x


class UseDDGCNet1(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DDGCNet1(num_classes=17, feature_dim=15)
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=17)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=17)
        self.train_miou = tm.JaccardIndex(task="multiclass", num_classes=17)
        self.val_miou = tm.JaccardIndex(task="multiclass", num_classes=17)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.view(B, N).float()
        pred = self.model(x, pos)
        pred = pred.transpose(2, 1)
        loss = self.loss(pred, y.long())
        self.train_acc(pred, y)
        self.train_miou(pred, y)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_miou", self.train_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.view(B, N).float()
        pred  = self.model(x, pos)
        pred = pred.transpose(2, 1)
        loss = self.loss(pred, y.long())
        self.val_acc(pred, y)
        self.val_miou(pred, y)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.view(B, N).float()
        pred = self.model(x, pos)
        pred = pred.transpose(2, 1)
        loss = self.loss(pred, y.long())
        self.val_acc(pred, y)
        self.val_miou(pred, y)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def predict_labels(self, data):
        with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
            with torch.no_grad():
                pos, x = data
                pos = pos.to(self.device)
                x = x.to(self.device)
                x = x.float()
                pred = self.model(x, pos)
                pred = pred.transpose(2, 1)
                pred = torch.argmax(pred, dim=1)
                return pred.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=60, gamma=0.5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            }
        }