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

def cache_cwt_data_from_dataloader(raw_dataloader: DataLoader, scales, fs, trim_ratio, output_file, num_workers=4):
    """
    从 DataLoader 中提取所有原始数据，并行计算 CWT 特征并缓存。
    """

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_file):
        print(f"✅ 缓存文件已存在: {output_file}。跳过计算。")
        return

    print("--- 步骤 1: 迭代 DataLoader 获取所有原始数据和标签 ---")
    all_raw_data = []
    all_labels = []

    # 迭代 DataLoader，将其中的所有 batch_size 数量的样本提取出来
    for raw_batch_data, raw_batch_labels in raw_dataloader:
        # raw_batch_data 形状可能为 [B, 1, L] 或 [B, L]
        # 转换为 NumPy，并展平批次维度以供多进程处理

        # 假设原始数据是 [B, 1, L]，先去掉维度 1
        raw_batch_data = raw_batch_data.squeeze(1)

        # 转换为 NumPy 数组
        all_raw_data.extend(raw_batch_data.cpu().numpy())
        all_labels.extend(raw_batch_labels.cpu().numpy())

        # 打印进度
        if len(all_labels) % (raw_dataloader.batch_size * 10) == 0:
            print(f"已提取 {len(all_labels)} 个样本...")

    print(f"DataLoader 迭代完毕，共获取 {len(all_labels)} 个样本。")

    # --- 步骤 2: 多进程并行计算 CWT ---
    print(f"\n--- 步骤 2: 开始使用 {num_workers} 个进程并行计算 CWT ---")

    cwt_func = partial(
        extract_cwt_features,
        scales=scales,
        fs=fs,
        trim_ratio=trim_ratio
    )

    with mp.Pool(processes=num_workers) as pool:
        # 使用 pool.imap_unordered 并行处理所有原始数据
        results_iterator = pool.imap_unordered(cwt_func, all_raw_data)

        processed_features = []

        # 逐个收集结果
        for i, features in enumerate(results_iterator):
            processed_features.append(features)

            if (i + 1) % 100 == 0 or (i + 1) == len(all_labels):
                print(f"已完成 {i + 1} 个样本的 CWT 计算...")

    print("所有样本 CWT 处理完成。")

    # --- 步骤 3: 保存结果 ---
    X_processed = np.stack(processed_features, axis=0)
    print(f"X_processed shape: {X_processed.shape}")
    Y_processed = np.array(all_labels)

    np.savez(output_file, X=X_processed, Y=Y_processed)
    print(f"\n🎉 特征和标签已成功缓存到 {output_file}。")


def create_dataloader_from_npz(
        npz_file_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        mean: Optional[float] = None,
        std: Optional[float] = None
) -> Tuple[DataLoader, float, float]:
    """
    从 npz 文件加载 CWT 特征，进行标准化，并创建 DataLoader。
    使用 torch.utils.data.TensorDataset 替换自定义 Dataset。

    Args:
        npz_file_path (str): 缓存的 .npz 文件路径。
        batch_size (int): DataLoader 的批次大小。
        shuffle (bool): 是否在每个 epoch 随机打乱数据。
        mean (Optional[float]): 用于标准化的均值。如果为 None，则自行计算。
        std (Optional[float]): 用于标准化的标准差。如果为 None，则自行计算。

    Returns:
        Tuple[DataLoader, float, float]:
            (DataLoader 实例, 实际使用的 mean, 实际使用的 std)
    """

    # 1. 加载数据并转换为 Tensor
    try:
        data = np.load(npz_file_path)
        # 将特征数据转换为 float32 并转为 Tensor
        X_tensor = torch.from_numpy(data['X'].astype(np.float32))
        # 将标签数据转换为 Long 类型 Tensor
        Y_tensor = torch.from_numpy(data['Y']).long()
        data.close()
    except FileNotFoundError:
        print(f"错误：找不到文件 {npz_file_path}")
        raise

    print(f"数据加载完成。样本总数: {X_tensor.shape[0]}，特征维度: {X_tensor.shape[1]}")

    # 2. 标准化 (Z-Score Normalization)
    if mean is None or std is None:
        # 如果未提供 mean 和 std，则计算并应用
        print("未提供 mean/std，正在计算并应用 Z-Score 标准化...")

        computed_mean = X_tensor.mean()
        computed_std = X_tensor.std()

        # 防止除零
        if computed_std == 0:
            computed_std = 1.0

        X_tensor = (X_tensor - computed_mean) / computed_std

        actual_mean = computed_mean.item()
        actual_std = computed_std.item()
    else:
        # 如果提供了 mean 和 std，则使用输入的值
        print(f"使用提供的 mean={mean:.4f}, std={std:.4f} 进行标准化...")

        X_tensor = (X_tensor - mean) / std

        actual_mean = mean
        actual_std = std

    # 3. 创建 TensorDataset 和 DataLoader
    # 使用 TensorDataset 直接包装特征和标签 Tensor
    dataset = TensorDataset(X_tensor, Y_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 特征已预处理，num_workers 设为 0
    )

    print(f"DataLoader 创建成功。")

    # 返回 DataLoader 和实际使用的 mean/std
    return dataloader, actual_mean, actual_std



# --- 示例：在主文件或预处理文件中调用 ---
if __name__ == '__main__':
    # 假设你已经定义了 MyRawDataset 和相关的参数
    from utils.ts_convertor import create_dataloader_from_arff
    from model.CWT import generate_adaptive_scales


    DATA_PATH = '/Users/hxh/PycharmProjects/final_thesis/Dataset/'
    DATA_NAME = 'AbnormalHeartbeat'


    TRAIN_FILE = f'{DATA_NAME}/{DATA_NAME}_TRAIN.arff'
    TEST_FILE = f'{DATA_NAME}/{DATA_NAME}_TEST.arff'

    train_path = DATA_PATH + TRAIN_FILE
    test_path = DATA_PATH + TEST_FILE


    train_loader, mean, std = create_dataloader_from_npz('/Users/hxh/PycharmProjects/final_thesis/Dataset/AbnormalHeartbeat/AbnormalHeartbeat_TEST.npz')


    train_dataloader, train_mean, train_std = create_dataloader_from_arff(
        arff_file_path=train_path, batch_size=64, shuffle=True
    )

    # 测试集：使用训练集的参数进行标准化
    test_dataloader, _, _ = create_dataloader_from_arff(
        arff_file_path=test_path, batch_size=64, shuffle=False,
        mean=train_mean, std=train_std
    )

    # --- 定义参数 ---
    L = 18305
    FS = 100.0
    TRIM_RATIO = 0.1
    SCALES = generate_adaptive_scales(L, num_scales=5)
    FILE_NAME = f'/{DATA_NAME}_cwt_features.npz'
    OUTPUT_FILE = DATA_PATH + DATA_NAME + FILE_NAME


    # 2. 运行缓存函数
    cache_cwt_data_from_dataloader(
        test_dataloader,
        SCALES,
        FS,
        TRIM_RATIO,
        OUTPUT_FILE,
        num_workers=10
    )

    # 3. 训练时使用 CachedCWTDataset 来读取缓存文件
    # from your_file import CachedCWTDataset # 假设你已定义
    # train_dataset = CachedCWTDataset(OUTPUT_FILE)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)