import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    """
    基础 1D 卷积模块，用于实现 Low/High Pass 滤波器。
    所有卷积核 K=2，无 padding。
    """

    def __init__(self, in_ch, out_ch, kernel_size=2, padding=0):
        super().__init__()
        # 使用 padding=0 和 stride=1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)
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
        self.low_pass_1 = Conv1DBlock(in_ch=input_channels, out_ch=3, kernel_size=7,padding=3)
        self.high_pass_1 = Conv1DBlock(in_ch=input_channels, out_ch=3, kernel_size=7,padding=3)

        # --- Stage 2 ---
        self.low_pass_2 = Conv1DBlock(in_ch=3, out_ch=3, kernel_size=5, padding=2)
        self.high_pass_2 = Conv1DBlock(in_ch=3, out_ch=3, kernel_size=5, padding=2)

        # --- Stage 3 ---
        self.low_pass_3 = Conv1DBlock(in_ch=3, out_ch=3, kernel_size=3, padding=1)
        self.high_pass_3 = Conv1DBlock(in_ch=3, out_ch=3, kernel_size=3, padding=1)

        self.n_classes = n_classes
        self.fc = None
        self._fc_initialized = False
        self.mlp = None

    def forward(self, x):
        # x 假设维度: (Batch, C_in=1, L_in)
        B = x.size(0)

        # --- Stage 1 ---
        L1 = self.low_pass_1(x)  # (B, 3, L_in-2)
        H1 = self.high_pass_1(x)  # (B, 3, L_in-2)

        # --- Stage 2 (应用于 H1) ---
        L2 = self.low_pass_2(L1)  # (B, 3, L_in-4)
        H2 = self.high_pass_2(L1)  # (B, 3, L_in-4)

        # --- Stage 3 ---
        L3 = self.low_pass_3(L2)
        H3 = self.high_pass_3(L2)


        # --- 扁平化并拼接 ---

        # 1. 扁平化 L1: (B, 3, L_in-2) -> (B, 3 * (L_in-2))
        feat_H2 = H2.view(B, -1)

        # 2. 扁平化 L2: (B, 3, L_in-4) -> (B, 3 * (L_in-4))
        # feat_L2 = L2.view(B, -1)

        # 3. 扁平化 H2: (B, 3, L_in-4) -> (B, 3 * (L_in-4))
        feat_H1 = H1.view(B, -1)

        feat_L3 = L3.view(B, -1)
        feat_H3 = H3.view(B, -1)

        # 拼接所有特征
        feat = torch.cat([feat_H2, feat_L3, feat_H1, feat_H3], dim=1)

        # 第一次调用时初始化 fc 层
        if not self._fc_initialized:
            self.fc = nn.Linear(feat.shape[1], self.n_classes)

            self.mlp = nn.Sequential(
                nn.Linear(feat.shape[1], 32),
                nn.ReLU(),
                nn.Linear(32, self.n_classes)
            )

            self._fc_initialized = True
            # 将 fc 层移到正确的设备上
            self.fc = self.fc.to(feat.device)

        # 分类
        logits = self.mlp(feat)
        return logits



# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = WaveletCNN(input_channels=1, n_classes=2)

    # 测试不同长度的输入
    print("测试不同长度的输入：")

    # 测试 1: 长度 100
    x1 = torch.randn(8, 1, 100)
    out1 = model(x1)
    print(f"输入形状: {x1.shape} -> 输出形状: {out1.shape}")

    # 测试 2: 长度 200
    model2 = WaveletCNN(input_channels=1, n_classes=2)
    x2 = torch.randn(8, 1, 200)
    out2 = model2(x2)
    print(f"输入形状: {x2.shape} -> 输出形状: {out2.shape}")

    # 验证参数
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters())}")