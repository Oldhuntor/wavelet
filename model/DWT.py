import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 定义 Haar 滤波器
# ==========================================
sqrt2 = torch.sqrt(torch.tensor(2.0))
h = torch.tensor([1/sqrt2, 1/sqrt2]).view(1, 1, -1)   # low-pass filter
g = torch.tensor([1/sqrt2, -1/sqrt2]).view(1, 1, -1)  # high-pass filter


# ==========================================
# 2. DWT 分解函数（支持任意层数）
# ==========================================
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


# ==========================================
# 3. 定义 MLP 模型类
# ==========================================
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


# ==========================================
# 4. 测试模型运行效果
# ==========================================
if __name__ == "__main__":
    # 假设你的时间序列长度为 N=128，batch size 为 B=16
    batch_size = 16
    seq_len     = 128

    x_input     = torch.randn(batch_size, 1, seq_len)

    # 创建模型，设置分解层数为参数，例如 level=3 或 level=4 等
    model       = DWT_MLP(input_length=seq_len,
                          levels=3,
                          hidden_dim=64,
                          output_dim=10)   # 比如用于10类分类任务

    y_pred      = model(x_input)

    print("Input shape:", x_input.shape)
    print("Output shape:", y_pred.shape)