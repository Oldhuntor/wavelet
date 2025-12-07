import torch
import torch.nn as nn
import numpy as np
import pywt

class WaveletMultiHeadClassifier(nn.Module):
    def __init__(self,
                 scales,
                 fs,
                 trim_ratio=0.1,
                 num_heads=4,
                 head_dim=64,
                 num_classes=3):
        super().__init__()

        self.scales = scales
        self.fs = fs
        self.trim_ratio = trim_ratio
        self.num_heads = num_heads
        self.head_dim = head_dim

        # 输入维度 = amplitude 和 phase 拼接后的 flatten
        # 但 flatten 大小依赖输入长度，所以这里用 lazy 模式
        self.projections = nn.ModuleList([
            nn.LazyLinear(head_dim) for _ in range(num_heads)
        ])

        self.classifier = nn.Linear(num_heads * head_dim, num_classes)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入信号，支持批处理。
                              形状: (B, N) 或 (N,)，其中 B 是批次大小，N 是信号长度。

        Returns:
            torch.Tensor: Logits，形状为 (B, num_classes)。
        """

        # --- 1. 处理输入形状和设备 ---

        # 确保输入至少是二维，便于批处理循环 (B, N)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (N,) -> (1, N)

        # ---------- 多头投影 ----------
        # 注意：如果 self.projections 是 nn.ModuleList，你需要使用 h_features 作为输入
        heads = [proj(x) for proj in self.projections]  # list of (B, head_dim)
        h = torch.cat(heads, dim=-1)  # (B, num_heads * head_dim)

        # ---------- 分类 ----------
        logits = self.classifier(h)  # (B, num_classes)
        if logits.dim() > 1 and 1 in logits.shape:
            logits = logits.squeeze()
        return logits


def generate_adaptive_scales(L, num_scales=5):
    """
    根据数据长度 L 自适应生成少量代表性尺度。

    Args:
        L (int): 信号或数据的长度 (N)。
        num_scales (int): 想要选择的尺度数量 (S)。

    Returns:
        numpy.ndarray: 包含 S 个尺度的数组。
    """
    # 最小周期/尺度: 3 (避免奈奎斯特和边界效应)
    a_min = 3.0

    # 最大周期/尺度: L / 2 (避免严重边界效应)
    a_max = L / 2.0

    if a_max <= a_min:
        # 如果 L 太小 (例如 L <= 6)，无法生成有效范围，返回一个默认值
        return np.array([1, 2])

    # 使用对数间隔生成尺度，保证分辨率平衡
    scales = np.logspace(np.log10(a_min), np.log10(a_max), num_scales)

    # 确保是整数类型 (虽然pywt接受浮点数，但整数更直观)
    # 也可以保持浮点数以获得精确的对数间隔
    return scales.astype(np.float32)



if __name__ == '__main__':
    scales = np.arange(10, 128)
    fs = 100

    model = WaveletMultiHeadClassifier(
        scales=scales,
        fs=fs,
        trim_ratio=0.1,
        num_heads=4,
        head_dim=64,
        num_classes=3
    )

    x = np.random.randn(2000)

    logits = model(x)
    print(logits)
