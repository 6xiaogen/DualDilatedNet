from torch import nn
import torch
import torchmetrics as tm
from models.layer import BasicPointLayer, DualEdgeGraphConvBlock, DualDilatedEdgeGraphConvBlock, ResidualBasicPointLayer, \
    PointFeatureImportance, STNkd
import lightning as L


class DDGCNet1(nn.Module):
    def __init__(self, num_classes=17, feature_dim=15):  # feature_dim=24
        """
        :param num_classes: Number of classes to predict
        """
        super(DDGCNet1, self).__init__()
        self.num_classes = num_classes

        self.stnkd = STNkd(k=15)   # feature_dim=24,要和feature_dim一样

        self.Dual_Dilated_block1 = DualEdgeGraphConvBlock(in_channels=feature_dim, hidden_channels=24,
                                                             out_channels=24,  # 第一次k_pos、k_dilated_pos必须为0
                                                             k=32, k_pos=0, k_dilated=12, k_dilated_pos=0,
                                                             # 两层嵌套图结构，第一层是k+k_pos,第二层k_dilated+k_dilated_pos
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

        self.mlp1 = BasicPointLayer(in_channels=380, out_channels=512)  # 380=60+120+200
        self.mlp2 = BasicPointLayer(in_channels=512, out_channels=1024)

        self.feature_importance = PointFeatureImportance(in_channels=1024)

        self.res_block1 = ResidualBasicPointLayer(in_channels=1024, out_channels=512, hidden_channels=512)
        self.res_block2 = ResidualBasicPointLayer(in_channels=512, out_channels=256, hidden_channels=256)

        self.out = BasicPointLayer(in_channels=256, out_channels=num_classes, is_out=1)  # is_out=True

    def forward(self, x, pos):
        # x [b,num,c]
        # x是所有的特征、pos是中心点的坐标
        cd = torch.cdist(pos, pos)  # 计算pos向量和pos向量之间的距离
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

        # # 对 x 进行最大池化，得到 [B, 1024]
        # x_max_pooled = torch.max(x, dim=1)[0]  # [2, 1024]

        # # 扩展维度，使得 x_max_pooled 的形状变为 [2, 1, 1024]
        # x_max_pooled = x_max_pooled.unsqueeze(1)  # [2, 1, 1024]

        # # 拼接 x_max_pooled 和 x1，得到 [2, 16000, 1104]
        # x1_expanded = x_max_pooled.expand(-1, x1.size(1), -1)  # 将 x_max_pooled 扩展为 [2, 16000, 1024]
        # x_concat = torch.cat([x1_expanded, x1], dim=2)  # 拼接 [2, 16000, 1024 + 80]

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
        B, N, C = x.shape     # x是16000个三角面片包含的24个参数的数据   ，每个面片中心的坐标，y是标签
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
                pos, x, y = data
                pos = pos.unsqueeze(0).to(self.device)
                x = x.unsqueeze(0).to(self.device)
                B, N, C = x.shape
                x = x.float()
                y = y.view(B, N).float()
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