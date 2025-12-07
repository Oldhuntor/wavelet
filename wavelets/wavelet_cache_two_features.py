import numpy as np
import torch
import pandas as pd
from sktime.datasets import load_from_arff_to_dataframe
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, Optional, Union
from sklearn.preprocessing import LabelEncoder # <-- 确保已导入！
import pywt
from utils import generate_adaptive_scales

def extract_cwt_features(x_single, scales, fs, trim_ratio):
    # ... (保持不变，用于单样本 CWT 计算)
    time_length = x_single.shape[0]
    coefficients, _ = pywt.cwt(
        x_single, scales, 'cmor1.5-1.0', sampling_period=1/fs
    )
    amplitude = np.abs(coefficients)
    phase = np.angle(coefficients)
    cut = int(time_length * trim_ratio)
    amplitude = amplitude[:, cut:-cut]
    phase = phase[:, cut:-cut]

    return  amplitude, phase


TRAIN_PATH = '/Users/hxh/PycharmProjects/final_thesis/Dataset/ECG/ECG5000/ECG5000_TRAIN.arff'
TEST_PATH = '/Users/hxh/PycharmProjects/final_thesis/Dataset/ECG/ECG5000/ECG5000_TEST.arff'  # 假设有测试集文件

# 1. 处理训练集 (
X_nested, y_labels = load_from_arff_to_dataframe(
    full_file_path_and_name=TRAIN_PATH,
    return_separate_X_and_y=True
)

n_instances = X_nested.shape[0]
n_channels = X_nested.shape[1]
sequence_length = X_nested.iloc[0, 0].shape[0]

X_np_3D = np.zeros((n_instances, n_channels, sequence_length), dtype=np.float32)

for i in range(n_instances):
    for j in range(n_channels):
        series_data = X_nested.iloc[i, j].to_numpy(dtype=np.float32)
        X_np_3D[i, j, :] = series_data

# 先标准化再做小波变换
mean = X_np_3D.mean()
std = X_np_3D.std()
X_np_scaled = (X_np_3D - mean) / std



