import numpy as np
import torch
import pandas as pd
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from sktime.datasets import load_from_arff_to_dataframe
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, Optional, Union
from sklearn.preprocessing import LabelEncoder # <-- 确保已导入！
from utils import generate_adaptive_scales, DATA_PATH, get_data_path, DATA_NAMES, get_save_path
import multiprocessing as mp
from functools import partial
import pywt
# from utils import load_mortlet_pt_dataloader


NUM_SCALES = 5
TRIM_RATIO = 0.05
FS = 1


def process_single_cwt(data_and_index: tuple, scales, fs, cut):
    """
    对单个样本执行 CWT、裁剪，并返回 (index, amplitude, phase)。
    """
    index, sample = data_and_index
    # print(index)

    # 执行 CWT
    coefficients, _ = pywt.cwt(
        sample, scales, 'cmor1.5-1.0', sampling_period=1 / fs
    )

    amplitude = np.abs(coefficients)
    phase = np.angle(coefficients)

    # 裁剪操作 (与你原代码一致)
    if cut > 0:
        amplitude = amplitude[:, cut:-cut]
        phase = phase[:, cut:-cut]


    return index, amplitude, phase  # 返回索引，保证后续能正确排序

def cache_morlet_coefficients(path:str, train_mean=None, train_std=None):
    # 1. 处理训练集 (
    X_nested, y_labels = load_from_arff_to_dataframe(
        full_file_path_and_name=path,
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
    if not train_mean:
        mean = X_np_3D.mean()
        std = X_np_3D.std()
        X_np_scaled = (X_np_3D - mean) / std
    else:
        X_np_scaled = (X_np_3D - train_mean) / train_std

    if y_labels.dtype != np.dtype('object') and np.issubdtype(y_labels.dtype, np.number):
        y_encoded = y_labels.astype(np.int64)
    else:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_labels)
        y_encoded = y_encoded.astype(np.int64)

    if y_encoded.min() >= 1:
        y_encoded = y_encoded - 1

    X_np_data = np.squeeze(X_np_scaled)
    num_samples = X_np_scaled.shape[0]
    sequence_length = X_np_scaled.shape[-1]
    scales = generate_adaptive_scales(sequence_length, num_scales=NUM_SCALES)
    # remove burnin period
    cut = int(sequence_length * TRIM_RATIO)
    indexed_input_data = list(enumerate(X_np_data))
    amplitude_array = np.empty((num_samples, NUM_SCALES, sequence_length - cut * 2), dtype=np.float32)
    phase_array = np.empty((num_samples, NUM_SCALES, sequence_length - cut * 2), dtype=np.float32)


    NUM_WORKERS = mp.cpu_count()  # 使用所有核心
    print(f"开始使用 {NUM_WORKERS} 个进程并行计算 CWT 特征...")



    partial_func = partial(
        process_single_cwt,
        scales=scales,
        fs=FS,
        cut=cut
    )

    # 使用 mp.Pool 执行
    with mp.Pool(processes=NUM_WORKERS) as pool:
        # pool.map 保证输出结果的顺序与输入 indexed_input_data 的顺序一致
        # 结果是一个包含 (index, amplitude, phase) 元组的列表
        results = list(pool.imap_unordered(partial_func, indexed_input_data))

    # --- 5. 收集结果并写入预分配数组 ---
    for index, amp, pha in results:
        amplitude_array[index] = amp
        phase_array[index] = pha
        print(index)

    return amplitude_array, phase_array, y_encoded, train_mean, train_std

if __name__ == '__main__':

    data_name = DATA_NAMES[-1]
    print(data_name)
    data_type = 'arff'
    train_path,test_path = get_data_path(DATA_PATH, data_name, data_type)
    amplitude_array, phase_array, y_encoded, mean, std = cache_morlet_coefficients(train_path)
    object_name = f'{data_name}_TRAIN.pt'

    save_path = get_save_path(DATA_PATH, data_name, object_name)

    torch.save({
        "amplitude": torch.from_numpy(amplitude_array),
        "phase": torch.from_numpy(phase_array),
        "labels": torch.from_numpy(y_encoded),
    }, save_path)

    object_name = f'{data_name}_TEST.pt'
    amplitude_array, phase_array, y_encoded, _, _ = cache_morlet_coefficients(test_path, mean, std)
    save_path = get_save_path(DATA_PATH, data_name, object_name)

    torch.save({
        "amplitude": torch.from_numpy(amplitude_array),
        "phase": torch.from_numpy(phase_array),
        "labels": torch.from_numpy(y_encoded),
    }, save_path)
