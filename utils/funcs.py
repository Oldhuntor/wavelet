import numpy as np
import pywt
import os
import multiprocessing as mp
from functools import partial
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple


def extract_cwt_features(x_single, scales, fs, trim_ratio):
    # ... (保持不变，用于单样本 CWT 计算)
    time_length = x_single.shape[0]
    coefficients, _ = pywt.cwt(
        x_single, scales, 'cmor1.5-1.0', sampling_period=1/fs
    )
    amplitude = np.abs(coefficients)
    phase = np.angle(coefficients)
    cut = int(time_length * trim_ratio)
    if cut > 0:
        amplitude = amplitude[:, cut:-cut]
        phase = phase[:, cut:-cut]
    features = np.concatenate([amplitude, phase], axis=0)
    return features.flatten()


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


