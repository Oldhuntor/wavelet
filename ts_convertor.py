import numpy as np
import torch
import pandas as pd
from sktime.datasets import load_from_arff_to_dataframe
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, Optional, Union
from sklearn.preprocessing import LabelEncoder # <-- 确保已导入！

def create_dataloader_from_arff(
        arff_file_path: str,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 4,
        mean: Optional[float] = None,
        std: Optional[float] = None
) -> Tuple[DataLoader, Optional[float], Optional[float]]:
    """
    加载 ARFF 时间序列文件，转换为 PyTorch Tensor，执行 Z-Score 标准化，并创建 DataLoader。
    同时，将标签 (1, 2, 3, 4, 5) 编码为 PyTorch 所需的 (0, 1, 2, 3, 4)。
    """

    # 1. 加载 ARFF 文件
    try:
        X_nested, y_labels = load_from_arff_to_dataframe(
            full_file_path_and_name=arff_file_path,
            return_separate_X_and_y=True
        )
        print(f"--- 成功加载文件: {arff_file_path} ---")

    except Exception as e:
        print(f"加载 ARFF 文件时发生错误: {e}")
        raise

    # 2. 手动将嵌套 DataFrame 转换为 3D NumPy 数组
    n_instances = X_nested.shape[0]
    n_channels = X_nested.shape[1]

    try:
        sequence_length = X_nested.iloc[0, 0].shape[0]
    except AttributeError:
        print("错误：X_nested 的单元格不是 pd.Series。请确认 ARFF 文件格式正确。")
        raise

    X_np_3D = np.zeros((n_instances, n_channels, sequence_length), dtype=np.float32)

    for i in range(n_instances):
        for j in range(n_channels):
            series_data = X_nested.iloc[i, j].to_numpy(dtype=np.float32)
            X_np_3D[i, j, :] = series_data

    # 3. Z-Score 标准化
    if mean is None or std is None:
        mean = X_np_3D.mean()
        std = X_np_3D.std()
        calculated_stats = True
    else:
        calculated_stats = False

    if std == 0:
        X_np_scaled = X_np_3D
    else:
        X_np_scaled = (X_np_3D - mean) / std

    print(f"标准化均值: {mean:.4f}, 标准差: {std:.4f}")


    if y_labels.dtype != np.dtype('object') and np.issubdtype(y_labels.dtype, np.number):
        # 如果已经是数字，则不需要编码
        y_encoded = y_labels.astype(np.int64)
    else:
        # 如果是字符串（如 'Abnormal'），则进行编码
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_labels)
        y_encoded = y_encoded.astype(np.int64)

    # 4. 转换为 PyTorch Tensor (注意标签修正!)

    # 关键修正：将标签 (1, 2, 3, 4, 5) 映射到 (0, 1, 2, 3, 4)
    # 因为 CrossEntropyLoss 要求标签从 0 开始
    # y_encoded = y_labels.astype(np.int64)
    if y_encoded.min() >= 1:
        # 假设最小标签是 1，进行映射
        y_tensor = torch.from_numpy(y_encoded - 1).long()
    else:
        # 否则，假设标签已经是 0 开始
        y_tensor = torch.from_numpy(y_encoded).long()

    X_tensor = torch.from_numpy(X_np_scaled).float()

    # 5. 创建 DataLoader
    dataloader = DataLoader(
        dataset=TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader, mean, std


if __name__ == '__main__':

    # --- 路径示例 ---
    TRAIN_PATH = '/Users/hxh/PycharmProjects/final_thesis/Dataset/ECG/ECG5000/ECG5000_TRAIN.arff'
    TEST_PATH = '/Users/hxh/PycharmProjects/final_thesis/Dataset/ECG/ECG5000/ECG5000_TEST.arff' # 假设有测试集文件

    # 1. 处理训练集 (计算并返回 mean/std)
    train_dataloader, train_mean, train_std = create_dataloader_from_arff(
        arff_file_path=TRAIN_PATH,
        batch_size=64,
        shuffle=True
    )

    print(f"\n训练集 DataLoader 创建完成，Batch 数: {len(train_dataloader)}")
    print(f"提取的训练集全局均值: {train_mean:.4f}, 标准差: {train_std:.4f}")

    # 2. 处理测试集 (使用训练集的 mean/std)
    test_dataloader, _, _ = create_dataloader_from_arff(
        arff_file_path=TEST_PATH,
        batch_size=64,
        shuffle=False,  # 测试集不应打乱
        mean=train_mean, # 使用训练集的参数
        std=train_std    # 使用训练集的参数
    )

    print(f"\n测试集 DataLoader 创建完成，Batch 数: {len(test_dataloader)}")

