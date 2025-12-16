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


# =====================================
# Simple Dataset for Demo
# =====================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data: numpy array of shape (num_samples, seq_length)
            labels: numpy array of shape (num_samples,)
        """
        self.data = torch.FloatTensor(data).unsqueeze(1)  # Add channel dim
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

