"""
每个类别训练多个小波 + ExtraTreesClassifier
========================================

思路:
  - 每个类别训练多个不同filter_length的小波
  - 用Daubechies初始化（更robust）
  - 所有类别的所有小波系数拼接
  - ExtraTreesClassifier分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import multiprocessing as mp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report



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


class LearnableWaveletDaubechies(nn.Module):
    """带Daubechies初始化 + QMF条件的可学习小波"""

    def __init__(self, filter_length, levels, use_frequency_constraint=True):
        super().__init__()
        self.filter_length = filter_length
        self.levels = levels
        self.use_frequency_constraint = use_frequency_constraint

        # 只训练low pass (Daubechies初始化)
        h0, h1 = get_daubechies_coefficients(filter_length)
        self.low_pass = nn.Parameter(torch.FloatTensor(h0))
        # self.high_pass = nn.Parameter(torch.FloatTensor(h1))

        # high_pass不是参数，用QMF条件计算
        # 注意：不要注册为parameter

    def get_high_pass(self):
        """用QMF条件从low_pass计算high_pass

        QMF条件: h1[n] = (-1)^n * h0[N-1-n]
        """
        N = self.filter_length
        high_pass = torch.zeros_like(self.low_pass)

        for n in range(N):
            high_pass[n] = ((-1) ** n) * self.low_pass[N - 1 - n]

        return high_pass

    def forward(self, x):
        """DWT分解"""
        coeffs = []
        current = x

        # 从low_pass计算high_pass
        high_pass = self.get_high_pass()

        for level in range(self.levels):
            # Pad
            pad_size = self.filter_length - 1
            current_padded = F.pad(current, (pad_size, pad_size), mode='constant', value=0)

            # Convolution
            low = F.conv1d(current_padded.unsqueeze(1),
                          self.low_pass.view(1, 1, -1),
                          stride=1).squeeze(1)
            high = F.conv1d(current_padded.unsqueeze(1),
                           high_pass.view(1, 1, -1),
                           stride=1).squeeze(1)

            # Downsample
            low = low[:, ::2]
            high = high[:, ::2]

            coeffs.append(high)  # detail
            current = low  # approx

        coeffs.append(current)  # final approx

        # Return in order: [approx_final, detail_N, ..., detail_1]
        return coeffs[::-1]

    def idwt_single(self, approx, detail, low_pass, high_pass, target_len):
        """Single level IDWT"""
        batch_size = approx.shape[0]
        max_len = max(approx.shape[1], detail.shape[1])

        # Upsample by inserting zeros
        up_approx = torch.zeros(batch_size, max_len * 2, device=approx.device)
        up_approx[:, ::2] = approx if approx.shape[1] == max_len else F.pad(approx, (0, max_len - approx.shape[1]))

        up_detail = torch.zeros(batch_size, max_len * 2, device=detail.device)
        up_detail[:, ::2] = detail if detail.shape[1] == max_len else F.pad(detail, (0, max_len - detail.shape[1]))

        # Reshape for conv1d
        up_approx = up_approx.unsqueeze(1)
        up_detail = up_detail.unsqueeze(1)
        low_pass_rec = low_pass.flip(0).view(1, 1, -1)
        high_pass_rec = high_pass.flip(0).view(1, 1, -1)

        # Convolve
        rec_approx = F.conv1d(up_approx, low_pass_rec, padding=self.filter_length - 1)
        rec_detail = F.conv1d(up_detail, high_pass_rec, padding=self.filter_length - 1)

        # Trim
        start = self.filter_length - 1
        min_len = min(rec_approx.shape[2], rec_detail.shape[2])
        reconstructed = (rec_approx[:, :, :min_len] + rec_detail[:, :, :min_len])[:, :, start:start + target_len]

        return reconstructed.squeeze(1)

    def idwt_multilevel(self, coeffs, signal_length):
        """Multi-level IDWT"""
        levels = len(coeffs) - 1
        current = coeffs[0]

        # 从low_pass计算high_pass
        high_pass = self.get_high_pass()

        for i in range(levels):
            detail = coeffs[i + 1]
            target_len = min(current.shape[1] * 2, signal_length)
            current = self.idwt_single(current, detail, self.low_pass, high_pass, target_len)

        return current[:, :signal_length]


class SingleClassWaveletModel:
    """单个类别的小波模型"""

    def __init__(self, filter_length, levels, signal_length):
        self.wavelet = LearnableWaveletDaubechies(
            filter_length, levels, use_frequency_constraint=True
        )
        self.signal_length = signal_length
        self.filter_length = filter_length
        self.levels = levels

    def train_wavelet(self, class_signals, target_loss=0.01, max_epochs=500,
                     lr=0.001, device='cpu', verbose=False):
        """训练到目标loss"""
        self.wavelet = self.wavelet.to(device)
        self.wavelet.train()

        optimizer = torch.optim.Adam(self.wavelet.parameters(), lr=lr)

        for epoch in range(max_epochs):
            optimizer.zero_grad()

            coeffs = self.wavelet(class_signals)
            reconstructed = self.wavelet.idwt_multilevel(coeffs, self.signal_length)

            loss = F.mse_loss(reconstructed, class_signals)

            loss.backward()
            optimizer.step()

            if verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

            if loss.item() <= target_loss:
                if verbose:
                    print(f"  达到目标 loss={target_loss} at epoch {epoch}")
                break

        # Freeze
        for param in self.wavelet.parameters():
            param.requires_grad = False
        self.wavelet.eval()

        return loss.item()

    def extract_features(self, signals, device='cpu'):
        """提取小波系数"""
        self.wavelet = self.wavelet.to(device)
        self.wavelet.eval()

        with torch.no_grad():
            coeffs = self.wavelet(signals)
            flat_coeffs = torch.cat([c.flatten(1) for c in coeffs], dim=1)

        return flat_coeffs.cpu().numpy()


def train_single_class_model(args):
    """Worker: 训练单个类别的单个小波"""
    class_i, model_idx, filter_length, levels, signal_length, class_signals, target_loss, device = args

    print(f"  [类别{class_i}, 模型{model_idx}] 训练 filter_length={filter_length}")

    model = SingleClassWaveletModel(filter_length, levels, signal_length)
    loss = model.train_wavelet(
        class_signals,
        target_loss=target_loss,
        max_epochs=500,
        device=device,
        verbose=False
    )

    print(f"  [类别{class_i}, 模型{model_idx}] 完成 loss={loss:.6f}")

    return class_i, model_idx, model, loss


class ClassSpecificMultiWaveletClassifier:
    """每个类别训练多个小波的分类器

    结构:
      类别0: [小波0(len=4), 小波1(len=8), 小波2(len=16), ...]
      类别1: [小波0(len=4), 小波1(len=8), 小波2(len=16), ...]
      类别2: [小波0(len=4), 小波1(len=8), 小波2(len=16), ...]
    """

    def __init__(self, filter_lengths, levels, signal_length, num_classes,
                 n_estimators=200, use_pca=True, pca_variance=0.95, n_jobs=None):
        """
        Args:
            filter_lengths: List of filter lengths, e.g. [4, 8, 16, 32]
            levels: DWT层数
            signal_length: 信号长度
            num_classes: 类别数
            n_estimators: ExtraTreesClassifier树数量
            use_pca: 是否用PCA降维
            pca_variance: PCA保留方差
            n_jobs: 并行任务数
        """
        self.filter_lengths = filter_lengths
        self.num_models_per_class = len(filter_lengths)
        self.levels = levels
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.n_jobs = n_jobs or mp.cpu_count()
        self.use_pca = use_pca
        self.pca_variance = pca_variance

        # 每个类别多个模型: models[class_i][model_idx]
        self.models = [[None for _ in range(self.num_models_per_class)]
                       for _ in range(num_classes)]

        self.pca = None
        self.classifier = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

        self.trained_wavelets = False
        self.trained_classifier = False

    def train_wavelets(self, train_loader, target_loss=0.01, device='cpu', verbose=True):
        """训练所有类别的所有小波"""
        if verbose:
            print("="*70)
            print("阶段1: 训练每个类别的多个小波 (Daubechies初始化)")
            print(f"每个类别 {self.num_models_per_class} 个小波")
            print(f"Filter lengths: {self.filter_lengths}")
            print(f"目标 loss: {target_loss}")
            print("="*70)

        # 按类别分组
        signals_by_class = [[] for _ in range(self.num_classes)]

        for signals, labels in train_loader:
            signals = signals.squeeze()
            for class_i in range(self.num_classes):
                mask = labels == class_i
                if mask.any():
                    signals_by_class[class_i].append(signals[mask])

        for class_i in range(self.num_classes):
            signals_by_class[class_i] = torch.cat(signals_by_class[class_i], dim=0)

        if verbose:
            for class_i in range(self.num_classes):
                print(f"类别 {class_i}: {len(signals_by_class[class_i])} 个样本")

        # 准备并行训练参数
        worker_args = []
        for class_i in range(self.num_classes):
            class_signals = signals_by_class[class_i].to(device)
            for model_idx, filter_length in enumerate(self.filter_lengths):
                worker_args.append((
                    class_i, model_idx, filter_length, self.levels,
                    self.signal_length, class_signals, target_loss, device
                ))

        if verbose:
            print(f"\n并行训练 {len(worker_args)} 个小波 (用 {self.n_jobs} workers)...")

        # 并行训练
        with mp.Pool(self.n_jobs) as pool:
            results = pool.map(train_single_class_model, worker_args)

        # 整理结果
        for class_i, model_idx, model, loss in results:
            self.models[class_i][model_idx] = model

        if verbose:
            print("\n" + "="*70)
            print("训练完成! Loss统计:")
            all_losses = [loss for _, _, _, loss in results]
            print(f"  平均: {np.mean(all_losses):.6f}")
            print(f"  标准差: {np.std(all_losses):.6f}")
            print(f"  范围: [{np.min(all_losses):.6f}, {np.max(all_losses):.6f}]")
            print("="*70)

        self.trained_wavelets = True

    def extract_all_features(self, signals, device='cpu'):
        """提取所有类别所有小波的系数"""
        all_features = []

        for class_i in range(self.num_classes):
            for model_idx in range(self.num_models_per_class):
                model = self.models[class_i][model_idx]
                features = model.extract_features(signals, device)
                all_features.append(features)

        # 拼接
        combined = np.hstack(all_features)
        return combined

    def train_classifier(self, train_loader, device='cpu', verbose=True):
        """训练ExtraTreesClassifier

        策略: 每个类别的小波只提取自己类别的系数
        例如:
          类别0的数据 → 用类别0的小波提取系数 → label 0
          类别1的数据 → 用类别1的小波提取系数 → label 1
        """
        if not self.trained_wavelets:
            raise RuntimeError("小波未训练!")

        if verbose:
            print("\n" + "="*70)
            print("阶段2: 训练ExtraTreesClassifier")
            print("策略: 每个类别的小波只提取自己类别的数据")
            print("="*70)

        # 按类别提取特征
        all_features = []
        all_labels = []

        for signals, labels in train_loader:
            signals = signals.squeeze()
            for class_i in range(self.num_classes):
                # 只处理这个类别的数据
                mask = labels == class_i
                if not mask.any():
                    continue

                class_signals = signals[mask]

                # 用这个类别的所有小波提取特征
                class_features = []
                for model_idx in range(self.num_models_per_class):
                    model = self.models[class_i][model_idx]
                    features = model.extract_features(class_signals, device)
                    class_features.append(features)

                # 拼接这个类别的所有小波特征
                combined_features = np.hstack(class_features)

                all_features.append(combined_features)
                all_labels.append(np.full(len(class_signals), class_i))

        X_train = np.vstack(all_features)
        y_train = np.concatenate(all_labels)
        print(all_labels)

        if verbose:
            print(f"原始特征: {X_train.shape}")
            print(f"  每个类别 {self.num_models_per_class} 个小波")
            for class_i in range(self.num_classes):
                count = (y_train == class_i).sum()
                print(f"  类别 {class_i}: {count} 个样本")

        # PCA降维
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_variance)
            X_train = self.pca.fit_transform(X_train)

            if verbose:
                print(f"PCA降维: {X_train.shape}")
                print(f"  保留方差: {self.pca.explained_variance_ratio_.sum():.3f}")

        # 训练分类器
        if verbose:
            print("\n训练ExtraTreesClassifier...")

        self.classifier.fit(X_train, y_train)
        print(f'xtrain shape {X_train.shape}')

        train_acc = self.classifier.score(X_train, y_train)

        if verbose:
            print(f"训练集准确率: {train_acc*100:.2f}%")
            print("✓ 训练完成")

        self.trained_classifier = True

    def predict(self, test_loader, device='cpu', method='vote'):
        """预测

        策略: 每个类别的小波独立预测，然后投票/平均

        Args:
            test_loader: 测试数据
            device: 'cpu' or 'cuda'
            method: 'vote' (投票) 或 'average' (平均概率)

        流程:
          测试样本 → 类别0小波 → 预测0
          测试样本 → 类别1小波 → 预测1
          测试样本 → 类别2小波 → 预测2
          → 取多数投票
        """
        if not self.trained_classifier:
            raise RuntimeError("分类器未训练!")

        all_predictions = []
        all_labels = []

        for batch in test_loader:
            if len(batch) == 2:
                signals, labels = batch
                all_labels.append(labels.numpy())
            else:
                signals = batch

            batch_size = len(signals)
            signals = signals.squeeze()

            # 每个类别的小波独立预测
            predictions_per_class = []  # [num_classes, batch_size]

            for class_i in range(self.num_classes):
                # 用这个类别的小波提取特征
                class_features = []
                for model_idx in range(self.num_models_per_class):
                    model = self.models[class_i][model_idx]
                    features = model.extract_features(signals, device)
                    class_features.append(features)

                # 拼接这个类别的所有小波特征
                combined = np.hstack(class_features)

                # PCA
                if self.use_pca:
                    combined = self.pca.transform(combined)

                # 预测
                print(f'predict combine shape {combined.shape}')
                preds = self.classifier.predict(combined)
                predictions_per_class.append(preds)

            # 转换为 [batch_size, num_classes]
            predictions_per_class = np.array(predictions_per_class).T  # [batch_size, num_classes]

            if method == 'vote':
                # 投票: 取众数
                from scipy import stats
                final_preds = stats.mode(predictions_per_class, axis=1, keepdims=False)[0]
            else:  # average
                # 平均: 取平均后round
                final_preds = np.round(predictions_per_class.mean(axis=1)).astype(int)

            all_predictions.append(final_preds)

        predictions = np.concatenate(all_predictions)

        if all_labels:
            true_labels = np.concatenate(all_labels)
            return predictions, true_labels
        else:
            return predictions, None

    def evaluate(self, test_loader, device='cpu', method='vote'):
        """评估

        Args:
            method: 'vote' (投票) 或 'average' (平均)
        """
        predictions, true_labels = self.predict(test_loader, device, method)

        if true_labels is None:
            raise ValueError("测试集没有标签!")

        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)

        return accuracy, report

    def train(self, train_loader, target_loss=0.01, device='cpu', verbose=True):
        """完整训练流程"""
        self.train_wavelets(train_loader, target_loss, device, verbose)
        self.train_classifier(train_loader, device, verbose)



# ============================================================================
# DataLoader处理工具
# ============================================================================

def squeeze_dataloader(loader):
    """
    Wrapper: [batch, 1, seq_len] -> [batch, seq_len]

    用法:
        train_loader = DataLoader(...)
        train_loader = squeeze_dataloader(train_loader)

        model.train(train_loader)  # 直接用
    """
    for batch in loader:
        if len(batch) == 2:
            signals, labels = batch
            if signals.dim() == 3 and signals.size(1) == 1:
                signals = signals.squeeze(1)
            yield signals, labels
        else:
            signals = batch
            if signals.dim() == 3 and signals.size(1) == 1:
                signals = signals.squeeze(1)
            yield signals


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 创建模型

    from utils import create_dataloader_from_arff
    from sklearn.metrics import confusion_matrix, accuracy_score
    from model.test import MultiModelParallelClassifier

    train_dataloader, train_mean, train_std = create_dataloader_from_arff(
        arff_file_path='/Users/hxh/PycharmProjects/wavelet/wavelet/Dataset/Computers/Computers_TRAIN.arff',
        batch_size=32,
        shuffle=True
    )
    # 测试集：使用训练集的参数进行标准化
    test_dataloader, _, _ = create_dataloader_from_arff(
        arff_file_path='/Users/hxh/PycharmProjects/wavelet/wavelet/Dataset/Computers/Computers_TEST.arff',
        batch_size=32,
        shuffle=False,
        mean=train_mean,
        std=train_std
    )

    model = ClassSpecificMultiWaveletClassifier(
        filter_lengths=[4, 8, 16, 32],  # 每个类别4个小波
        levels=3,
        signal_length=720,
        num_classes=2,
        n_estimators=200,
        use_pca=True,
        pca_variance=0.95
    )

    # 训练
    model.train(train_dataloader, target_loss=0.01, device='cpu')

    # 预测
    predictions, labels = model.predict(test_dataloader)

    # 评估
    acc, report = model.evaluate(test_dataloader)
    print(f"准确率: {acc * 100:.2f}%")
    print(report)
