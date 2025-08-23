import torch
from torch import nn

def fps(xyz, npoint: int):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    distance = distance.float().to(device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1).float().to(device)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn(x, k=16):
    x_t = x.transpose(2, 1)
    pairwise_distance = torch.cdist(x_t, x_t, p=2)
    idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx


def knn_dual(x, x_pos, k, k_pos = 0):
    x_t = x.transpose(2, 1)
    pairwise_distance = torch.cdist(x_t, x_t, p=2)
    idx_feature = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]

    if k_pos is 0:
        idx = idx_feature
    else:
        x_pos_t = x_pos.transpose(2, 1)
        pairwise_distance_pos = torch.cdist(x_pos_t, x_pos_t, p=2)
        idx_pos = pairwise_distance_pos.topk(k=k_pos + 1, dim=-1, largest=False)[1][:, :, 1:]
        idx = torch.cat((idx_feature, idx_pos), dim=2)

    return idx


def batched_index_select(input, dim:int, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def get_graph_feature(x, k:int =20, idx=None, pos=None, edge_function: int = 200):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if pos is None:
            idx = knn(x, k=k)
        else:
            idx = knn(pos, k=k)
    device = x.device

    idx_org = idx

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if edge_function == 249:
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx_org

def get_dualgraph_feature(x, pos, k: int = 20, k_pos: int = 0, idx=None, farther_bool:bool=False, edge_function: int = 200):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if farther_bool is False:
        if k_pos is 0:
            idx = knn_dual(pos, pos, k=k, k_pos=k_pos)
        else:
            idx = knn_dual(x, pos, k=k, k_pos=k_pos)
    device = x.device

    idx_org = idx

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k+k_pos, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k+k_pos, 1)

    if edge_function == 249:
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx_org


class DualEdgeGraphConvBlock(nn.Module):
    def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, edge_function: int, k: int = 32,  k_pos: int =0, k_dilated: int =8, k_dilated_pos: int =6 ):
        super(DualEdgeGraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        self.in_channels2 = in_channels
        self.k = k
        self.k_pos = k_pos
        self.k_dilated = k_dilated
        self.k_dilated_pos = k_dilated_pos
        if edge_function == 249:
            self.in_channels = self.in_channels * 2
            self.in_channels2 = self.in_channels * 2

        self.convdilated = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, pos, idx=None):
        x_t = torch.transpose(x, 2, 1)    # x_t （1，24，16000）
        # if pos is None:
        #     pos = torch.clone(x)
        pos_t = torch.transpose(pos, 2, 1)

        out2, idx = get_dualgraph_feature(x_t, pos=pos_t, k=self.k_dilated, k_pos=self.k_dilated_pos,  idx=idx, farther_bool=False, edge_function=self.edge_function)
        out3 = self.convdilated(out2)
        out4 = torch.max(out3, dim=-1, keepdim=False)[0]

        out5, idx2 = get_dualgraph_feature(out4, pos=pos_t, k=self.k, k_pos=self.k_pos, idx=idx,farther_bool=False, edge_function=self.edge_function)
        out6 = self.conv(out5)
        out7 = torch.max(out6, dim=-1, keepdim=False)[0]

        out = torch.transpose(out7, 2, 1)
        return out, idx2



class DualDilatedEdgeGraphConvBlock(nn.Module):
    def __init__(self, in_channels:int, hidden_channels:int, out_channels: int, edge_function: int, all_k: int =128, k: int=32, k_pos:int=0, k_dilated:int=8, k_dilated_pos: int =6):
        super(DualDilatedEdgeGraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        if all_k < k:
            raise ValueError(f'Dilation k {all_k} must be larger than k {k}')
        self.all_k = all_k
        self.k_pos = k_pos
        self.k = k
        self.k_dilated_pos = k_dilated_pos
        self.k_dilated = k_dilated

        if edge_function==249:
            self.in_channels = self.in_channels * 2
            self.in_channels2 = self.in_channels * 2

        self.convdilated = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )


    def forward(self, x, pos, cd=None):
        x_t = torch.transpose(x, 2, 1)
        pos_t = torch.transpose(pos, 2, 1)
        B, N, C = x.shape
        if cd is None:
            cd = torch.cdist(pos, pos, p=2)
        all_k = min(self.all_k, N)
        idx_l = torch.topk(cd, all_k, largest=False)[1]
        idx_l = torch.reshape(idx_l, (B * N, -1))
        pos=torch.reshape(pos,(B * N, -1))
        idx_fps = fps(pos[idx_l], self.k+self.k_pos).long()
        for ii in range(1, len(idx_l.shape)):
            if ii != 1:
                idx_fps = idx_fps.unsqueeze(ii)
        expanse = list(idx_l.shape)
        expanse[0] = -1
        expanse[1] = -1
        idx_fps = idx_fps.expand(expanse)
        idx_fps = torch.gather(idx_l, 1, idx_fps)
        idx_fps = torch.reshape(idx_fps, (B, N, -1))

        out2, idx = get_dualgraph_feature(x_t, pos=pos_t, k=self.k_dilated, k_pos=self.k_dilated_pos, idx=idx_fps,farther_bool=False, edge_function=self.edge_function)  # farther_bool=False那么idx=idx_fp没有用了
        out3 = self.convdilated(out2)
        out4 = torch.max(out3, dim=-1, keepdim=False)[0]

        out5, idx2 = get_dualgraph_feature(out4, pos=pos_t, k=self.k, k_pos=self.k_pos, idx=idx_fps, farther_bool=True, edge_function=self.edge_function)
        out6 = self.conv(out5)
        out7 = torch.max(out6, dim=-1, keepdim=False)[0]

        out = torch.transpose(out7, 2, 1)
        return out, idx


class BasicPointLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float=0.1, is_out: int = 0):
        super(BasicPointLayer, self).__init__()
        if is_out == 1:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.conv(x).transpose(2, 1)


class ResidualBasicPointLayer(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float=0.1):
        super(ResidualBasicPointLayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        if in_channels != out_channels:
            self.rescale = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout)
            )
        else:
            self.rescale = nn.Identity()

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.conv(x).transpose(2, 1) + self.rescale(x).transpose(2, 1)

class PointFeatureImportance(nn.Module):

    def __init__(self, in_channels: int):
        super(PointFeatureImportance, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels, track_running_stats=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.conv(x.transpose(2, 1))
        return x * weight.transpose(2, 1)

class PointnetModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super(PointnetModule, self).__init__()

        self.layer1 = BasicPointLayer(in_channels=in_channels, out_channels=64, dropout=dropout)
        self.layer2 = BasicPointLayer(in_channels=64, out_channels=128, dropout=dropout)
        self.layer3 = BasicPointLayer(in_channels=128, out_channels=256, dropout=dropout)

        self.layer4 = BasicPointLayer(in_channels=256, out_channels=out_channels, dropout=dropout, is_out=1)

        self.fc = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_channels)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        global_features = torch.max(x, 1, keepdim=False)[0]

        point_features = self.layer4(x).transpose(1, 2)

        global_features = global_features.unsqueeze(2).expand(-1, -1, x.size(1))
        combined_features = torch.cat([point_features, global_features], dim=1)

        output = self.fc(combined_features.transpose(1, 2))
        return output


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        x = x.transpose(2, 1)
        x_org = x
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=torch.float32).flatten().view(1, self.k * self.k).repeat(batchsize, 1)

        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        trans_x = x.view(-1, self.k, self.k)

        x_org = x_org.transpose(2, 1)
        x_org = torch.bmm(x_org, trans_x)
        return x_org