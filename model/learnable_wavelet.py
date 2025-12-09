import torch
import torch.nn as nn
import torch.nn.functional as F

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
