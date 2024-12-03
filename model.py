import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet


class PlaningNetwork(nn.Module):
    def __init__(self, M, num_pts):
        super().__init__()
        self.M = M  # 预测的路径数量
        self.num_pts = num_pts  # 每条路径的点数
        self.backbone = EfficientNet.from_pretrained('efficientnet-b2', in_channels=6)  # 使用 EfficientNet 提取图像特征

        use_avg_pooling = False  # TODO
        if use_avg_pooling:
            self.plan_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局平均池化
                nn.Flatten(),  # 将特征展平为一维向量
                nn.BatchNorm1d(1408),  # 归一化
                nn.ReLU(),  # 激活函数
                nn.Linear(1408, 4096),  # 全连接层，隐藏层大小为 4096
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, M * (num_pts * 3 + 1))  # 输出路径预测和分类
            )
        else:
            self.plan_head = nn.Sequential(
                nn.BatchNorm2d(1408),  # 归一化通道维度
                nn.Conv2d(1408, 32, 1),  # 通道降维，1x1卷积
                nn.BatchNorm2d(32),
                nn.Flatten(),
                nn.ELU(),  # 激活函数
                nn.Linear(1024, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, M * (num_pts * 3 + 1))  # 输出
            )

    def forward(self, x):
        features = self.backbone.extract_features(x)  # 提取特征
        raw_preds = self.plan_head(features)  # 路径预测
        pred_cls = raw_preds[:, :self.M]  # 分类输出
        pred_trajectory = raw_preds[:, self.M:].reshape(-1, self.M, self.num_pts, 3)  # 路径点的三维位置

        # 路径点的后处理，确保数值稳定性
        pred_xs = pred_trajectory[:, :, :, 0:1].exp()  # 将 x 坐标变换到正值范围
        pred_ys = pred_trajectory[:, :, :, 1:2].sinh()  # y 坐标允许为正负
        pred_zs = pred_trajectory[:, :, :, 2:3]  # z 坐标直接输出
        return pred_cls, torch.cat((pred_xs, pred_ys, pred_zs), dim=3)  # 返回分类和处理后的路径


class SequencePlanningNetwork(nn.Module):
    def __init__(self, M, num_pts):
        super().__init__()
        self.M = M  # M 表示输出类别的数量（例如类别数）
        self.num_pts = num_pts  # num_pts 表示轨迹点的数量
        self.backbone = EfficientNet.from_pretrained('efficientnet-b2', in_channels=6)
        # 使用预训练的 EfficientNet-b2 作为特征提取器，输入通道数是 6，通常表示多个传感器的输入（如图像和深度图像）。

        # 计划头部（plan_head）：负责对 backbone 提取的特征进行进一步处理
        self.plan_head = nn.Sequential(
            # 特征维度从（6, 450, 800）转换为（1408, 4, 8）后进行处理
            nn.BatchNorm2d(1408),  # 批量归一化，针对 1408 个通道进行归一化
            nn.Conv2d(1408, 32, 1),  # 卷积层，卷积核大小为 1，输出 32 个通道，特征尺寸（4, 8）
            nn.BatchNorm2d(32),  # 对 32 个通道进行批量归一化
            nn.Flatten(),  # 将多维的输出拉平为一维，以便进行全连接处理
            nn.ELU(),  # ELU 激活函数，用于增加非线性
        )

        # GRU 层，用于处理时间序列数据。GRU（门控循环单元）用于捕捉时序信息
        self.gru = nn.GRU(input_size=1024, hidden_size=512, bidirectional=True, batch_first=True)  # bidirectional=True 表示双向 GRU

        # 计划头部的后续部分（plan_head_tip），用于生成最终的轨迹预测
        self.plan_head_tip = nn.Sequential(
            nn.Flatten(),  # 拉平特征
            nn.ELU(),  # 激活函数
            nn.Linear(1024, 4096),  # 全连接层，将输入特征映射到 4096 维
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(4096, M * (num_pts * 3 + 1))  # 最后一层输出：M * (num_pts * 3 + 1)，多出的 1 可能是分类任务的类别（cls）
        )

    def forward(self, x, hidden):
        # 前向传播过程
        features = self.backbone.extract_features(x)  # 从输入 x 中提取特征，x 可能是一个多模态输入（如图像和深度）

        # 通过 plan_head 处理提取的特征
        raw_preds = self.plan_head(features)
        # 将特征输入到 GRU 中，GRU 会处理时序数据
        raw_preds, hidden = self.gru(raw_preds[:, None, :], hidden)  # GRU 输入数据形状：N, L, H_in（batch_first=True）
        # 使用 plan_head_tip 进一步处理 GRU 的输出，生成最终的预测
        raw_preds = self.plan_head_tip(raw_preds)

        # 预测结果分为两个部分：预测类别（cls）和预测轨迹
        pred_cls = raw_preds[:, :self.M]  # 类别预测，大小为 M
        pred_trajectory = raw_preds[:, self.M:].reshape(-1, self.M, self.num_pts, 3)  # 轨迹预测，形状为 [batch_size, M, num_pts, 3]

        # 将三维坐标的每一维进行变换
        pred_xs = pred_trajectory[:, :, :, 0:1].exp()  # 对 x 坐标进行指数变换
        pred_ys = pred_trajectory[:, :, :, 1:2].sinh()  # 对 y 坐标进行双曲正弦变换
        pred_zs = pred_trajectory[:, :, :, 2:3]  # z 坐标直接使用（没有变换）

        # 返回类别预测、轨迹预测和更新后的隐藏状态
        return pred_cls, torch.cat((pred_xs, pred_ys, pred_zs), dim=3), hidden


class AbsoluteRelativeErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon  # 为避免除零错误，设置一个小的常量 epsilon，默认值为 1e-4

    def forward(self, pred, target):
        # 计算预测值和真实值之间的相对误差
        error = (pred - target) / (target + self.epsilon)
        # 返回误差的绝对值
        return torch.abs(error)


class SigmoidAbsoluteRelativeErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon  # 为避免除零错误，设置一个小的常量 epsilon，默认值为 1e-4

    def forward(self, pred, target):
        # 计算预测值和真实值之间的相对误差
        error = (pred - target) / (target + self.epsilon)
        # 使用 Sigmoid 函数压缩误差范围
        return torch.sigmoid(torch.abs(error))


class MultipleTrajectoryPredictionLoss(nn.Module):
    def __init__(self, alpha, M, num_pts, distance_type='angle'):
        super().__init__()
        self.alpha = alpha  # TODO: 当前没有使用的参数
        self.M = M  # 预测的轨迹数量
        self.num_pts = num_pts  # 每条轨迹的点数

        self.distance_type = distance_type  # 距离度量类型，默认为 'angle'
        if self.distance_type == 'angle':
            self.distance_func = nn.CosineSimilarity(dim=2)  # 使用余弦相似度作为距离度量
        else:
            raise NotImplementedError  # 如果使用其他类型的距离度量，抛出错误

        # 分类损失函数
        self.cls_loss = nn.CrossEntropyLoss()
        # 回归损失函数，使用 SmoothL1Loss
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        # 可替代的回归损失（当前未启用）
        # self.reg_loss = SigmoidAbsoluteRelativeErrorLoss()
        # self.reg_loss = AbsoluteRelativeErrorLoss()

    def forward(self, pred_cls, pred_trajectory, gt):
        """
        pred_cls: [B, M]  # 预测的类别（即选择的轨迹）
        pred_trajectory: [B, M * num_pts * 3]  # 预测的轨迹数据，每条轨迹的每个点包含 3 个坐标（x, y, z）
        gt: [B, num_pts, 3]  # 真实的轨迹数据，每条轨迹有 num_pts 个点，每个点包含 3 个坐标（x, y, z）
        """
        assert len(pred_cls) == len(pred_trajectory) == len(gt)  # 确保输入的 batch 大小一致

        # 重塑预测的轨迹数据：将其形状从 [B, M * num_pts * 3] 转为 [B, M, num_pts, 3]
        pred_trajectory = pred_trajectory.reshape(-1, self.M, self.num_pts, 3)

        with torch.no_grad():
            # 第一步：计算预测轨迹和真实轨迹之间的距离
            # 选择每条轨迹的最后一个点作为轨迹的结束位置
            pred_end_positions = pred_trajectory[:, :, self.num_pts-1, :]  # [B, M, 3]
            gt_end_positions = gt[:, self.num_pts-1:, :].expand(-1, self.M, -1)  # [B, 1, 3] -> [B, M, 3]

            # 计算结束点之间的余弦相似度，距离越小，两个向量越相似
            distances = 1 - self.distance_func(pred_end_positions, gt_end_positions)  # [B, M]

            # 找到与真实轨迹最相似的预测轨迹索引
            index = distances.argmin(dim=1)  # [B]

        # 根据最小距离的索引选择最接近的预测轨迹
        gt_cls = index
        pred_trajectory = pred_trajectory[torch.tensor(range(len(gt_cls)), device=gt_cls.device), index, ...]  # [B, num_pts, 3]

        # 计算分类损失：预测类别与真实类别之间的损失
        cls_loss = self.cls_loss(pred_cls, gt_cls)

        # 计算回归损失：预测轨迹与真实轨迹之间的损失
        reg_loss = self.reg_loss(pred_trajectory, gt).mean(dim=(0, 1))  # 对每个样本的损失取平均

        return cls_loss, reg_loss



if __name__ == '__main__':
    # model = EfficientNet.from_pretrained('efficientnet-b2', in_channels=6)
    model = PlaningNetwork(M=3, num_pts=20)

    dummy_input = torch.zeros((1, 6, 256, 512))

    # features = model.extract_features(dummy_input)
    features = model(dummy_input)

    pred_cls = torch.rand(16, 5)
    pred_trajectory = torch.rand(16, 5*20*3)
    gt = torch.rand(16, 20, 3)

    loss = MultipleTrajectoryPredictionLoss(1.0, 5, 20)

    loss(pred_cls, pred_trajectory, gt)
