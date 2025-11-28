import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from typing import Tuple

class LearnableWaveletLayer(nn.Module):
    def __init__(self, kernel_size=4):
        super().__init__()
        sqrt2 = torch.sqrt(torch.tensor(2.0))
        h_init = torch.tensor([1/sqrt2, 1/sqrt2])
        g_init = torch.tensor([1/sqrt2, -1/sqrt2])

        # 可学习参数
        self.h = nn.Parameter(h_init.view(1, 1, kernel_size))
        self.g = nn.Parameter(g_init.view(1, 1, kernel_size))

    def forward(self, x):
        A_next = F.conv1d(x, self.h, stride=2)
        D_next = F.conv1d(x, self.g, stride=2)
        return A_next, D_next


class LearnableWaveletLayerDau(nn.Module):
    """
    可学习的小波分解层，针对单通道时间序列 (C_in=1)。
    根据 kernel_size 初始化为对应的 Daubechies (Db) 正交小波系数。
    """

    def __init__(self, kernel_size=4, num_levels=3):
        super().__init__()
        self.num_levels = num_levels

        # 创建多个可学习小波层
        self.wavelet_layers = nn.ModuleList([
            LearnableWaveletLayer(kernel_size)
            for _ in range(num_levels)
        ])

    def forward(self, x):
        # x 形状: (Batch, 1, Length)

        features = []
        A_current = x

        # 逐级分解
        for i in range(self.num_levels):
            layer = self.wavelet_layers[i]

            # 仅在低频近似系数 A 上继续分解
            A_next, D_current = layer(A_current)

            # 将当前尺度的细节系数 (D) 视为特征
            features.append(D_current)

            # 更新下一级的输入为 A_next
            A_current = A_next

        # 最后一级的近似系数 A 也是重要的低频特征
        features.append(A_current)

        # 将所有尺度的特征 (D1, D2, D3, A3) 拼接起来
        # 注意：它们具有不同的长度 (L/2, L/4, L/8, L/8)
        return features


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 确保输入 x 的形状是 (Batch, 1, Length)
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"Input must be single-channel (shape: Bx1xL), received {x.shape}")

        # 卷积操作：
        # - weight=self.g/self.h (形状: 1x1xL)
        # - stride=2 实现降采样 (DWT)

        # 填充策略：使用 'circular' 或 'reflect' 填充来模拟小波的边界处理
        # 默认为 0 填充 ('valid' 或 'same' 依赖于 L 和 kernel_size)

        # 计算填充量，以实现 'same' 卷积效果（保持分解后长度 L/2）
        padding = self.kernel_size // 2 - 1

        # 1. 低频近似 A (Approximation)
        A_next = F.conv1d(
            input=x,
            weight=self.g,
            stride=2,
            padding=padding
            # DWT 标准库通常使用周期性或反射性边界，这里简化为 0 填充
        )

        # 2. 高频细节 D (Detail)
        D_next = F.conv1d(
            input=x,
            weight=self.h,
            stride=2,
            padding=padding
        )

        return A_next, D_next

def dwt_learnable(x, wavelet_layer: LearnableWaveletLayer, levels=3):
    A_current = x.clone()
    features = []

    for level in range(levels):
        A_next, D_next = wavelet_layer(A_current)
        features.append(D_next)
        A_current = A_next

    features.append(A_current)   # 最后一层低频部分也加入特征中

    feats_flattened = [f.flatten(start_dim=1) for f in features]
    return torch.cat(feats_flattened, dim=1)


class LWT(nn.Module):
    def __init__(self, input_length: int,
                 levels: int = 3,
                 hidden_dim: int = 64,
                 output_dim: int = 1):
        super().__init__()

        self.wavelet_layer = LearnableWaveletLayer(kernel_size=2)
        self.levels = levels

        # 用一次前向传播确定 DWT 后的特征维度大小
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, input_length)
            feat_dummy = dwt_learnable(dummy_x, wavelet_layer=self.wavelet_layer, levels=self.levels)
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
        feats = dwt_learnable(x.float(), wavelet_layer=self.wavelet_layer, levels=self.levels)
        out = self.mlp(feats)
        return out
