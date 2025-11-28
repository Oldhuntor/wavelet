import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_convertor import create_dataloader_from_arff

class TimeSeriesMLP(nn.Module):
    """
    一个用于时间序列分类的简单多层感知器 (MLP) 模型。
    """

    def __init__(self, input_dim, num_classes, hidden_size=512, dropout_rate=0.5):
        super(TimeSeriesMLP, self).__init__()

        # 1. 展平层 (Flattening Layer)
        # 将输入数据 (Channels, Length) 展平为单个长向量 (Channels * Length)
        self.flatten = nn.Flatten()

        # 2. 线性层 (Linear Layers)
        # 第一层：从输入维度到隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_size)
        # 第二层：隐藏层到另一个隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        # 第三层（输出层）：从隐藏层到类别数
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

        # 3. Dropout 层 (用于正则化，防止过拟合)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x 形状: (Batch, Channels, Length)

        # 1. 展平: (Batch, Channels, Length) -> (Batch, Channels * Length)
        x = self.flatten(x)

        # 2. 第一层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 3. 第二层
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 4. 输出层（不加激活函数，因为 CrossEntropyLoss 会自动处理 softmax）
        logits = self.fc3(x)

        return logits



# ==========================================
# 1. 定义 Haar 滤波器
# ==========================================
sqrt2 = torch.sqrt(torch.tensor(2.0))
h = torch.tensor([1/sqrt2, 1/sqrt2]).view(1, 1, -1)   # low-pass filter
g = torch.tensor([1/sqrt2, -1/sqrt2]).view(1, 1, -1)  # high-pass filter


def dwt_haar(x, levels=3):
    """
    对输入信号 x (shape: [B, C, L]) 做 Haar 小波分解。
    参数 levels 控制分解层数。
    返回所有层的近似和细节系数组合后的特征张量。
    """
    A_current = x.clone()
    features = []

    for level in range(levels):
        A_next = F.conv1d(A_current, h.to(x.device), stride=2)
        D_next = F.conv1d(A_current, g.to(x.device), stride=2)

        # 保存当前层的特征（可以拼接或展平）
        features.append(D_next)
        A_current = A_next

    # 最后一层的低频部分也加入特征中
    features.append(A_current)

    # 拼接所有尺度的特征为一个向量
    feats_flattened = [f.flatten(start_dim=1) for f in features]
    return torch.cat(feats_flattened, dim=1)


class DWT_MLP(nn.Module):
    def __init__(self, input_length: int, levels: int = 3,
                 hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.levels = levels

        # 用一次前向传播确定 DWT 后的特征维度大小
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, input_length)
            feat_dummy = dwt_haar(dummy_x, levels=self.levels)
            self.feature_dim = feat_dummy.shape[1]

        # 定义简单 MLP 网络结构
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        输入 x shape: [batch_size, channels=1, length]
        输出为预测值，例如分类或回归结果。
        """
        feats = dwt_haar(x.float(), levels=self.levels)
        out = self.mlp(feats)
        return out

