import numpy as np
import pandas as pd
import pywt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- 1. 参数配置 ---
# 假设信号采样率为 100 Hz
SAMPLING_RATE = 100
# 信号长度 (例如 1秒, 100个采样点)
SIGNAL_LENGTH = 100
# CWT分析的尺度范围。尺度对应于频率，需要根据数据特性调整。
# 较小的尺度对应高频，较大的尺度对应低频。
SCALES = np.arange(1, 51)


# --- 2. Morlet 小波特征提取函数 ---
def morlet_feature_extractor(signal: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    使用 Morlet 小波进行连续小波变换 (CWT)，并提取统计特征。

    Args:
        signal: 1D 时间序列信号 (shape: [SIGNAL_LENGTH,])
        scales: CWT 分析的尺度数组

    Returns:
        1D 特征向量 (特征维度 = 尺度数 * 2)
    """
    # 1. 执行 CWT (使用 'morl' 即 Morlet 小波)
    # coefs 的 shape: [scales_len, signal_length]
    coefs, frequencies = pywt.cwt(signal, scales, 'morl', sampling_period=1 / SAMPLING_RATE)

    # 2. 提取特征：对每个尺度 (频率) 的小波系数进行统计
    # CWT 系数是复数，我们取其模 (magnitude) 来代表能量
    coefs_abs = np.abs(coefs)

    # 特征 1: 每个尺度的平均能量 (Mean)
    mean_features = np.mean(coefs_abs, axis=1)

    # 特征 2: 每个尺度的能量波动 (Standard Deviation)
    std_features = np.std(coefs_abs, axis=1)

    # 3. 合并特征并展平为 1D 向量
    # 特征向量: [mean_s1, mean_s2, ..., std_s1, std_s2, ...]
    feature_vector = np.hstack([mean_features, std_features])

    return feature_vector


# --- 3. 模拟时间序列数据 (Binary Classification) ---
def generate_data(num_samples=200):
    """
    模拟 200 个时间序列样本，分为两类 (0 和 1)。
    类别 0: 主要是低频信号 (Normal)
    类别 1: 包含高频冲击信号 (Fault)
    """
    X = []
    y = []
    t = np.linspace(0, 1, SIGNAL_LENGTH, endpoint=False)

    for i in range(num_samples):
        # 类别 0: 低频正弦波 + 噪声
        if i < num_samples / 2:
            signal = 0.5 * np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.1, SIGNAL_LENGTH)
            label = 0
        # 类别 1: 高频正弦波 + 噪声 + 冲击 (模拟故障)
        else:
            signal = 0.8 * np.sin(2 * np.pi * 25 * t) + np.random.normal(0, 0.2, SIGNAL_LENGTH)
            # 增加一个高频冲击
            signal += np.exp(-(t - 0.5) ** 2 / 0.01) * 2
            label = 1

        X.append(signal)
        y.append(label)

    return np.array(X), np.array(y)


# --- 4. 主程序执行 ---
if __name__ == "__main__":

    # 1. 模拟数据
    X_raw, y = generate_data()
    print(f"原始信号数据形状: {X_raw.shape}")

    # 2. 特征提取 (Morlet CWT)
    print("\n--- 开始 Morlet 特征提取 ---")

    # 初始化特征列表
    X_features = []
    for i, signal in enumerate(X_raw):
        features = morlet_feature_extractor(signal, SCALES)
        X_features.append(features)

    X_processed = np.array(X_features)

    # 验证特征维度: 尺度数 (50) * 统计量数 (2) = 100
    print(f"提取后的特征数据形状: {X_processed.shape}")
    print(f"特征维度 (每个样本): {X_processed.shape[1]}")

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")

    # 4. 训练 XGBoost 分类器
    print("\n--- 开始训练 XGBoost 模型 ---")

    # 初始化 XGBoost 分类器
    # 使用一些默认的优秀参数
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',  # 二分类问题
        n_estimators=100,  # 迭代次数
        use_label_encoder=False,  # 避免 sklearn 版本警告
        eval_metric='logloss',  # 评估指标
        random_state=42
    )

    # 训练模型
    xgb_clf.fit(X_train, y_train)

    # 5. 模型评估
    y_pred = xgb_clf.predict(X_test)
    y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]

    print("\n--- 模型评估结果 ---")
    print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告 (Classification Report):")
    print(classification_report(y_test, y_pred))

    # 6. 特征重要性分析 (可选)
    # 查看哪些尺度/频率对分类最重要
    importances = xgb_clf.feature_importances_

    # 尺度数
    num_scales = len(SCALES)

    # 平均特征的重要性 (前 num_scales 个)
    mean_importances = importances[:num_scales]
    # 标准差特征的重要性 (后 num_scales 个)
    std_importances = importances[num_scales:]

    # 找出最重要的 3 个尺度（基于平均能量）
    top_3_indices = np.argsort(mean_importances)[-3:][::-1]
    top_3_scales = SCALES[top_3_indices]

    # 将尺度近似转换为频率 (仅为方便解释，记住是伪频率)
    # Morlet中心频率 fc 约等于 0.8125 Hz (pywt 默认值)
    morlet_fc = pywt.cwt_freq(1, 'morl', 1 / SAMPLING_RATE)[0]  # 确定 Morlet 的 fc
    top_3_freqs = morlet_fc / top_3_scales

    print("\n--- Morlet 特征重要性分析 ---")
    print(f"原始信号中最关键的 3 个特征尺度 (Scale): {top_3_scales}")
    print(f"近似对应的中心频率 (Hz): {[f'{f:.2f}Hz' for f in top_3_freqs]}")

    # 在本例中，最重要的尺度应该对应于高频信号 (类别 1 的 25Hz)