import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 1D 卷积模块 (基础构建块)
# ----------------------------
class Conv1DBlock(nn.Module):
    """
    基础 1D 卷积模块，用于实现 Low/High Pass 滤波器。
    所有卷积核 K=2，无 padding。
    """

    def __init__(self, in_ch, out_ch, kernel_size=2):
        super().__init__()
        # 使用 padding=0 和 stride=1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)
        self.act = nn.ReLU()  # 添加激活函数

    def forward(self, x):
        return self.act(self.conv(x))


# ----------------------------
# 主模型：两级 1D 分解结构
# ----------------------------
class WaveletCNN(nn.Module):
    def __init__(self, input_channels=1, n_classes=2):
        super().__init__()

        # --- Stage 1 ---
        self.low_pass_1 = Conv1DBlock(in_ch=input_channels, out_ch=3, kernel_size=2)
        self.high_pass_1 = Conv1DBlock(in_ch=input_channels, out_ch=3, kernel_size=2)

        # --- Stage 2 ---
        self.low_pass_2 = Conv1DBlock(in_ch=3, out_ch=3, kernel_size=2)
        self.high_pass_2 = Conv1DBlock(in_ch=3, out_ch=3, kernel_size=2)

        self.fc = nn.LazyLinear(n_classes)

    def forward(self, x):
        # x 假设维度: (Batch, C_in=1, L_in)
        B = x.size(0)

        # --- Stage 1 ---
        L1 = self.low_pass_1(x)  # (B, 3, L_in)
        H1 = self.high_pass_1(x)  # (B, 3, L_in)

        # --- Stage 2 (应用于 H1) ---
        L2 = self.low_pass_2(H1)  # (B, 3, L_in)
        H2 = self.high_pass_2(H1)  # (B, 3, L_in)

        # --- 扁平化并拼接 ---

        # 1. 扁平化 L1: (B, 3, L_in) -> (B, 3 * L_in)
        feat_L1 = L1.view(B, -1)

        # 2. 扁平化 L2: (B, 3, L_in) -> (B, 3 * L_in)
        feat_L2 = L2.view(B, -1)

        # 3. 扁平化 H2: (B, 3, L_in) -> (B, 3 * L_in)
        feat_H2 = H2.view(B, -1)

        # 拼接所有特征 (总维度: B, 9 * L_in)
        feat = torch.cat([feat_L1, feat_L2, feat_H2], dim=1)

        # 分类
        logits = self.fc(feat)
        return logits


# ----------------------------
# 示例使用
# ----------------------------
# ----------------------------
# 示例使用
# ----------------------------
if __name__ == '__main__':
    # 假设 Batch Size = 16, 输入序列长度 L_in = 15
    L_in = 15
    B = 16
    input_tensor = torch.randn(B, 1, L_in)

    # 实例化模型时必须传入 input_len
    model = WaveletCNN(input_channels=1, n_classes=10, input_len=L_in)

    # 运行前向传播
    output, features = model(input_tensor)

    print(f"输入形状: {input_tensor.shape}")
    print(f"输出 Logits 形状: {output.shape}")
    print(f"拼接特征形状: {features.shape}")
