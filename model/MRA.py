import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import load_morlet_pt, MorletDataset

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 为了保持拼接后的通道数可控，我们将每个分支的输出通道设为 out_channels // 4
        # 剩下一个分支留给 1x1 卷积或者直接分配
        branch_channels = out_channels // 4

        # 分支 1: 小感受野 (Kernel=3) - 捕捉高频/局部细节
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )

        # 分支 2: 中感受野 (Kernel=5)
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )

        # 分支 3: 大感受野 (Kernel=7) - 捕捉低频/全局趋势
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )

        # 分支 4: 1x1 卷积 - 保持特征并变换通道
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels - 3 * branch_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels - 3 * branch_channels),
            nn.ReLU()
        )

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        b1 = self.branch1(x)
        # 在通道维度 (dim=1) 拼接
        return torch.cat([b3, b5, b7, b1], dim=1)


# =====================================
# 2. Residual Block with Skip Connection
#    (ResNet-style)
# =====================================
class MRA_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 主路径：真正的多尺度分析
        self.multiscale_path = MultiScaleConv(in_channels, out_channels)

        # 捷径 (Skip Connection)
        # 如果输入输出通道数不同，或者需要改变维度，使用 1x1 卷积进行投影
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()  # 直接通过，不做修改

        self.relu = nn.ReLU()

    def forward(self, x):
        # F(x)
        out = self.multiscale_path(x)

        # x (经过可能的投影)
        residual = self.shortcut(x)

        # Output = F(x) + x
        out = out + residual
        out = self.relu(out)
        return out


# =====================================
# 3. Dual-branch Classifier
# =====================================
class DualFeatureMRAClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # --- Amplitude Branch ---
        self.branch_amp = nn.Sequential(
            # Layer 1: 输入5 -> 输出32 (升维)
            MRA_ResBlock(5, 32),
            nn.MaxPool1d(2),  # 下采样，减少计算量

            # Layer 2: 输出32 -> 输出64 (加深特征)
            MRA_ResBlock(32, 64),
            nn.AdaptiveAvgPool1d(1)  # Global Pooling -> (B, 64, 1)
        )

        # --- Phase Branch ---
        self.branch_pha = nn.Sequential(
            MRA_ResBlock(5, 32),
            nn.MaxPool1d(2),

            MRA_ResBlock(32, 64),
            nn.AdaptiveAvgPool1d(1)
        )

        # --- Fusion & Classification ---
        # 两个分支各输出 64，拼接后为 128
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(64, num_classes)
        )

    def forward(self, amp, pha):
        # Amp forward
        f1 = self.branch_amp(amp)  # (B, 64, 1)
        f1 = f1.view(f1.size(0), -1)  # Flatten -> (B, 64)

        # Phase forward
        f2 = self.branch_pha(pha)  # (B, 64, 1)
        f2 = f2.view(f2.size(0), -1)  # Flatten -> (B, 64)

        # Concatenate
        x = torch.cat([f1, f2], dim=1)  # (B, 128)

        return self.fc(x)

