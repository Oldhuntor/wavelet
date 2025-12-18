import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


# =====================================
# Multi-Scale Convolution Module
# =====================================
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        branch_channels = out_channels // 4

        # Branch 1: Small receptive field (Kernel=3)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )

        # Branch 2: Medium receptive field (Kernel=5)
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )

        # Branch 3: Large receptive field (Kernel=7)
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )

        # Branch 4: 1x1 convolution
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
        return torch.cat([b3, b5, b7, b1], dim=1)


# =====================================
# MRA Residual Block
# =====================================
class MRA_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.multiscale_path = MultiScaleConv(in_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.multiscale_path(x)
        residual = self.shortcut(x)
        out = out + residual
        out = self.relu(out)
        return out


# =====================================
# Single Branch Time Series Classifier
# =====================================
class MRATimeSeriesClassifier(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        """
        Args:
            input_channels: Number of input channels (1 for univariate time series)
            num_classes: Number of output classes
            seq_length: Length of input time series
        """
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # Layer 1: 1 -> 32 channels
            MRA_ResBlock(input_channels, 32),
            nn.MaxPool1d(2),  # seq_length / 2

            # Layer 2: 32 -> 64 channels
            MRA_ResBlock(32, 64),
            nn.MaxPool1d(2),  # seq_length / 4

            # Layer 3: 64 -> 128 channels
            MRA_ResBlock(64, 128),
            nn.AdaptiveAvgPool1d(1)  # Global pooling -> (B, 128, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_length)
        features = self.feature_extractor(x)  # (B, 128, 1)
        features = features.view(features.size(0), -1)  # Flatten to (B, 128)
        output = self.classifier(features)
        return output

if __name__ == '__main__':
    #
    # from torchviz import make_dot
    # import torch
    #
    # # 创建模型示例
    # model = MRATimeSeriesClassifier(input_channels=1, num_classes=2)
    #
    # # 创建虚拟输入
    # batch_size = 2
    # seq_length = 100
    # x = torch.randn(batch_size, 1, seq_length)
    #
    # # 生成计算图
    # y = model(x)
    # dot = make_dot(y, params=dict(model.named_parameters()))
    # dot.render('mra_classifier', format='png', cleanup=True)
    # print("计算图已保存为 mra_classifier.png")
    import torch
    import onnx
    import os

    # 创建模型并导出为ONNX格式
    model = MRATimeSeriesClassifier(input_channels=1, num_classes=10)
    batch_size = 1
    seq_length = 100
    x = torch.randn(batch_size, 1, seq_length)

    # 导出ONNX文件
    torch.onnx.export(
        model,
        x,
        "mra_classifier.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}},
        opset_version=11
    )

    print("ONNX文件已保存为 mra_classifier.onnx")
    print("请访问 https://netron.app/ 上传此文件进行可视化")