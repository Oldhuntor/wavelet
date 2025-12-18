import numpy as np
import pywt
import os
import multiprocessing as mp
from functools import partial
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
from utils.constant import *
from utils.objects import MorletDataset
import matplotlib.pyplot as plt

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


def load_morlet_pt(path):
    data = torch.load(path)
    amp = data["amplitude"].numpy()
    pha = data["phase"].numpy()
    y = data["labels"].numpy()
    return amp, pha, y


def get_data_path(data_path:str, data_name:str, data_type:str):
    train_name = data_path + f'{data_name}/{data_name}_TRAIN.{data_type}'
    test_name = data_path + f'{data_name}/{data_name}_TEST.{data_type}'
    return train_name, test_name

def get_save_path(data_path, data_name:str, object_name):
    path = data_path + f'{data_name}/{object_name}'
    return path


def load_mortlet_pt_dataloader(data_name):
    TRAIN_FILE, TEST_FILE = get_data_path(DATA_PATH, data_name, 'pt')
    amp_train, pha_train, y_train = load_morlet_pt(TRAIN_FILE)
    amp_test, pha_test, y_test = load_morlet_pt(TEST_FILE)
    # ---------- Build Dataset & DataLoader ----------
    train_ds = MorletDataset(amp_train, pha_train, y_train)
    test_ds = MorletDataset(amp_test, pha_test, y_test)
    return train_ds, test_ds


def get_daubechies_coefficients(filter_length: int) -> tuple:
    """
    Generate Daubechies wavelet coefficients using PyWavelets.

    Args:
        filter_length: Length of the filter (must be even, typically 2, 4, 6, 8, ...)
                      This determines the Daubechies order: db1, db2, db3, etc.

    Returns:
        h0: Low-pass filter coefficients (scaling function / father wavelet)
        h1: High-pass filter coefficients (mother wavelet)
    """
    if filter_length % 2 != 0:
        raise ValueError("Filter length must be even")

    db_order = filter_length // 2
    wavelet_name = f'db{db_order}'

    # Get wavelet object
    wavelet = pywt.Wavelet(wavelet_name)

    # Get decomposition filters: dec_lo (low-pass), dec_hi (high-pass)
    h0 = np.array(wavelet.dec_lo, dtype=float)
    h1 = np.array(wavelet.dec_hi, dtype=float)

    return h0, h1

