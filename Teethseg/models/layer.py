import torch
from torch import nn


def fps(xyz, npoint: int):
    """
    Farthest Point Sampling (FPS) algorithm for selecting a subset of points from a point cloud.

    Args:
        xyz (torch.Tensor): Input point cloud tensor of shape (B, N, C), where B is the batch size, N is the number of points, and C is the number of dimensions.
        npoint (int): Number of points to select.

    Returns:
        torch.Tensor: Tensor of shape (B, npoint) containing the indices of the selected points.
    """
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
    """
    Performs k-nearest neighbors (knn) search on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_points, num_dims).
        k (int): Number of nearest neighbors to find.

    Returns:
        torch.Tensor: Index tensor of shape (batch_size, num_points, k), containing the indices of the k nearest neighbors for each point.
    """

    x_t = x.transpose(2, 1)   # x_t[1,24,16000]
    pairwise_distance = torch.cdist(x_t, x_t, p=2) # [1,16000,16000]  # 计算点集中两两之间的距离
    idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


def knn_dual(x, x_pos, k, k_pos = 0):
    """
    #  将原本的kkn改为dual，也就是通过坐标k_pos得到的和通过特征k得到的一块拼接了,另外第一次卷积k_pos是0，且特征就是坐标
    Performs k-nearest neighbors (knn) search on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_points, num_dims).
        k (int): Number of nearest neighbors to find.  动态点个数
        x_pos 是坐标
        k_pos 是静态点个数
    Returns:
        torch.Tensor: Index tensor of shape (batch_size, num_points, k), containing the indices of the k nearest neighbors for each point.
    """

    x_t = x.transpose(2, 1)  # x_t[1,24,16000]
    pairwise_distance = torch.cdist(x_t, x_t, p=2)  # [1,16000,16000]  # 计算点集中两两之间的特征距离
    idx_feature = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]  # (batch_size, num_points, k)

    if k_pos is 0:
        idx = idx_feature
    else:
        x_pos_t = x_pos.transpose(2, 1)
        pairwise_distance_pos = torch.cdist(x_pos_t, x_pos_t, p=2)  # 计算点集中两两之间的几何距离
        idx_pos = pairwise_distance_pos.topk(k=k_pos + 1, dim=-1, largest=False)[1][:, :, 1:]  # (batch_size, num_points, k)
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

    idx_org = idx  # idx [1,16000,32] # 1，16000，32

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()   # num_dis 24

    x = x.transpose(2, 1).contiguous()   # 1，16000，24

    feature = x.view(batch_size * num_points, -1)[idx, :]   # 通过每个点的索引，找到对应点的24个特征，  512000，24
    feature = feature.view(batch_size, num_points, k, num_dims)   # 1，16000，32，24
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # 1，16000，32，24，将24复制展开32行

    if edge_function == 249:
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()   # 1，48，16000，32，这48个通过全连接去输出新的通道

    return feature, idx_org   # feature就是新的数据，维度不是24了，而是48



# 修改后的get_graph_feature，对应的是对偶图结构
def get_dualgraph_feature(x, pos, k: int = 20, k_pos: int = 0, idx=None, farther_bool:bool=False, edge_function: int = 200):  # k_pos判断是不是第一次进行动态卷积，如果是第一次就把x特征改为坐标去弄
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if farther_bool is False:
        if k_pos is 0:
            idx = knn_dual(pos, pos, k=k, k_pos=k_pos)
        else:
            idx = knn_dual(x, pos, k=k, k_pos=k_pos)
    device = x.device

    # 接下来需要通过get_graphs_dilated去得到每个点扩张聚合的特征

    idx_org = idx   # 1，16000，32

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()   # num_dis 24

    x = x.transpose(2, 1).contiguous()   # 1，16000，24

    feature = x.view(batch_size * num_points, -1)[idx, :]   # 通过每个点的索引，找到对应点的24个特征，  512000，24
    feature = feature.view(batch_size, num_points, k+k_pos, num_dims)   # 1，16000，32，24
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k+k_pos, 1)  # 1，16000，32，24，将24复制展开32行

    if edge_function == 249:
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()   # 1，48，16000，32，这48个通过全连接去输出新的通道

    return feature, idx_org   # feature就是新的数据，维度不是24了，而是48


# 修改EdgeGraphConvBlock
class DualEdgeGraphConvBlock(nn.Module):
    """
    EdgeGraphConvBlock is a module that performs edge graph convolution on input features.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        edge_function (int): Type of edge function to use. Can be "global", "local", or "local_global".,oringin is str
        k (int): Number of nearest neighbors to consider for local edge function. Default is 32.

    Raises:
        ValueError: If edge_function is not one of "global", "local", or "local_global".

    Attributes:
        edge_function (int): Type of edge function used.
        in_channels (int): Number of input channels.
        k (int): Number of nearest neighbors considered for local edge function.
        conv (nn.Sequential): Sequential module consisting of convolutional layers.

    """

    def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, edge_function: int, k: int = 32,  k_pos: int =0, k_dilated: int =8, k_dilated_pos: int =6 ):
        super(DualEdgeGraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        self.in_channels2 = in_channels   # 经过依次构建图结构，特征就要乘以2
        self.k = k
        self.k_pos = k_pos
        self.k_dilated = k_dilated
        self.k_dilated_pos = k_dilated_pos
        if edge_function == 249:   # 是249，说明经过图结构，24就变成48，输入通道要*2
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
        """
        Forward pass of the EdgeGraphConvBlock.

        Args:
            x (torch.Tensor): Input features.
            idx (torch.Tensor, optional): Index tensor for graph construction of shape (B, N, K), where B is the batch size. Defaults to None.
            pos (torch.Tensor, optional): Position tensor of shape (B, N, D), where D is the number of dimensions. Default is None.

        Returns:
            torch.Tensor: Output features after edge graph convolution.
            torch.Tensor: Updated index tensor.

        """

        x_t = torch.transpose(x, 2, 1)    # x_t （1，24，16000）
        # if pos is None:
        #     pos = torch.clone(x)
        pos_t = torch.transpose(pos, 2, 1)

        # 先进行扩张卷积聚合
        out2, idx = get_dualgraph_feature(x_t, pos=pos_t, k=self.k_dilated, k_pos=self.k_dilated_pos,  idx=idx, farther_bool=False, edge_function=self.edge_function)  # farther_bool是否是获取更远特征
        out3 = self.convdilated(out2)   # out3：B，24，16000，32，这里24就不是三角面片的几何信息，不过这里和论文不一致，论文是48
        out4 = torch.max(out3, dim=-1, keepdim=False)[0]   # 1，24，16000

        # 再进行二次聚合得到更加详细的特征
        out5, idx2 = get_dualgraph_feature(out4, pos=pos_t, k=self.k, k_pos=self.k_pos, idx=idx,farther_bool=False, edge_function=self.edge_function)
        out6 = self.conv(out5)
        out7 = torch.max(out6, dim=-1, keepdim=False)[0]       # 1，24，16000

        out = torch.transpose(out7, 2, 1)
        return out, idx2



class DualDilatedEdgeGraphConvBlock(nn.Module):
    """
    A block implementing a dilated edge graph convolution operation.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        edge_function (int): Type of edge function to use. Must be one of "global", "local", or "local_global".
        dilation_k (int): Number of nearest neighbors to consider for the dilation operation.
        k (int): Number of nearest neighbors to consider for the graph convolution operation.

    Raises:
        ValueError: If `dilation_k` is smaller than `k` or if `edge_function` is not one of the allowed values.

    all_k是扩展的所有特征点，k是通过最远点采样（在all_k中）得到的32个点，all_k_pos是扩展的所有坐标几何点，k_pos是通过最远点采样（在all_k_pos中）得到的点，形成dual
    """

    def __init__(self, in_channels:int, hidden_channels:int, out_channels: int, edge_function: int, all_k: int =128, k: int=32, k_pos:int=0, k_dilated:int=8, k_dilated_pos: int =6):
        super(DualDilatedEdgeGraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        if all_k < k:
            raise ValueError(f'Dilation k {all_k} must be larger than k {k}')
        self.all_k = all_k
        self.k_pos = k_pos  # 并没有参与更远特征的图结构，也就是没有dual
        self.k = k
        self.k_dilated_pos = k_dilated_pos   # # 并没有参与更远特征的图结构，也就是没有dual
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
        """
        Forward pass of the dilated edge graph convolution block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size, N is the number of nodes,
                and C is the number of input channels.
            pos (torch.Tensor, optional): Position tensor of shape (B, N, D), where D is the number of dimensions.
                Defaults to None.
            cd (torch.Tensor, optional): Pairwise distance tensor of shape (B, N, N). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, N, out_channels), where out_channels is the number of output channels.
            torch.Tensor: Index tensor of shape (B, N, K), representing the indices of the nearest neighbors.

        """
        # 构建更远特征的knn，这里是以坐标构建的，然后给到get_graph去构建图结构
        x_t = torch.transpose(x, 2, 1)    # x_t:[1,60,16000]  #
        pos_t = torch.transpose(pos, 2, 1)
        B, N, C = x.shape
        if cd is None:      # cd:[1,16000,16000]
            cd = torch.cdist(pos, pos, p=2)
        all_k = min(self.all_k, N)
        idx_l = torch.topk(cd, all_k, largest=False)[1]   # 16000，200
        idx_l = torch.reshape(idx_l, (B * N, -1))
        pos=torch.reshape(pos,(B * N, -1))
        idx_fps = fps(pos[idx_l], self.k+self.k_pos).long()   # idx_fps:[16000,32]  # 16000，32

        # 好像是为了改变维度，又好像没用
        for ii in range(1, len(idx_l.shape)):
            if ii != 1:
                idx_fps = idx_fps.unsqueeze(ii)
        expanse = list(idx_l.shape)
        expanse[0] = -1
        expanse[1] = -1    # [-1,-1]
        idx_fps = idx_fps.expand(expanse)
        idx_fps = torch.gather(idx_l, 1, idx_fps)
        idx_fps = torch.reshape(idx_fps, (B, N, -1))   # 输出：[1,16000,32]

        # 先进行扩张卷积聚合
        out2, idx = get_dualgraph_feature(x_t, pos=pos_t, k=self.k_dilated, k_pos=self.k_dilated_pos, idx=idx_fps,farther_bool=False, edge_function=self.edge_function)  # farther_bool=False那么idx=idx_fp没有用了
        out3 = self.convdilated(out2)  # out3：1，24，16000，32，这里24就不是三角面片的几何信息，不过这里和论文不一致，论文是48
        out4 = torch.max(out3, dim=-1, keepdim=False)[0]  # 1，24，16000

        # 再进行二次聚合得到更加详细的特征
        out5, idx2 = get_dualgraph_feature(out4, pos=pos_t, k=self.k, k_pos=self.k_pos, idx=idx_fps, farther_bool=True, edge_function=self.edge_function)
        out6 = self.conv(out5)
        out7 = torch.max(out6, dim=-1, keepdim=False)[0]  # 1，24，16000

        out = torch.transpose(out7, 2, 1)
        return out, idx




class BasicPointLayer(nn.Module):
    """
    Basic point layer consisting of a 1D convolution, batch normalization, leaky ReLU, and dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float=0.1, is_out: int = 0):
        """
        Initializes the BasicPointLayer.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param dropout: Dropout probability. Default is 0.1.
        """
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
    """
    Basic point layer consisting of a 1D convolution, batch normalization, leaky ReLU, and dropout
    with a residual connection.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float=0.1):
        """
        Initializes the BasicPointLayer.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param dropout: Dropout probability. Default is 0.1.
        """
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
            self.rescale = nn.Identity()    # 一个恒等映射，即它不会对输入数据进行任何变换

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.conv(x).transpose(2, 1) + self.rescale(x).transpose(2, 1)   # 两个分支的结果相加，形成输出。这种残差连接的思想是借鉴自 ResNet，能够让深层模型更稳定地进行训练

# 主要用于加权特征的方式
class PointFeatureImportance(nn.Module):
    """
    Point Feature Importance module.
    """

    def __init__(self, in_channels: int):
        """
        Initializes the PointFeatureImportance module.
        :param in_channels: Number of input channels.
        """
        super(PointFeatureImportance, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels, track_running_stats=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.conv(x.transpose(2, 1))
        return x * weight.transpose(2, 1)


class STNkd(nn.Module):
    """
    STNkd module.
    """

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