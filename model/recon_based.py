import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

try:
    from model.adaptive_filters import LearnableWaveletTransform
except ImportError:
    print("需要 wavelet_classifier.py 中的 LearnableWaveletTransform")
    raise


class SingleWaveletReconstructor:
    """单个类别的小波重构器"""

    def __init__(self, filter_length, levels, signal_length):
        self.wavelet = LearnableWaveletTransform(
            filter_length, levels, 'random',
            use_frequency_constraint=True,
            use_learnable_activation=False
        )
        self.signal_length = signal_length
        self.filter_length = filter_length
        self.levels = levels
        self.best_loss = float('inf')

    def train_to_target_loss(self, class_signals, target_loss=0.01, max_epochs=500,
                             lr=0.001, device='cpu', verbose=False):
        """训练到目标loss

        Args:
            class_signals: (N, signal_length) 某个类别的所有信号
            target_loss: 目标loss，达到就停止
            max_epochs: 最大训练轮数
            lr: 学习率
            device: 'cpu' or 'cuda'
            verbose: 是否打印进度
        """
        self.wavelet = self.wavelet.to(device)
        self.wavelet.train()

        optimizer = torch.optim.Adam(self.wavelet.parameters(), lr=lr)

        for epoch in range(max_epochs):
            optimizer.zero_grad()

            # 前向传播
            class_signals = class_signals.squeeze()
            coeffs = self.wavelet(class_signals)
            reconstructed = self.wavelet.idwt_multilevel(coeffs, self.signal_length)

            # 重构损失
            loss = F.mse_loss(reconstructed, class_signals)

            # 反向传播
            loss.backward()
            optimizer.step()

            if verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

            # 达到目标就停止
            if loss.item() <= target_loss:
                if verbose:
                    print(f"  达到目标 loss={target_loss} at epoch {epoch}")
                break

        # 冻结参数
        for param in self.wavelet.parameters():
            param.requires_grad = False
        self.wavelet.eval()

        final_loss = loss.item()
        self.best_loss = final_loss

        return final_loss

    def train_on_class(self, class_signals, max_epochs=100, lr=0.001,
                       device='cpu', verbose=False):
        """训练这个小波来重构某个类别的信号（早停版本）

        Args:
            class_signals: (N, signal_length) 某个类别的所有信号
            max_epochs: 最大训练轮数
            lr: 学习率
            device: 'cpu' or 'cuda'
            verbose: 是否打印进度
        """
        self.wavelet = self.wavelet.to(device)
        self.wavelet.train()

        optimizer = torch.optim.Adam(self.wavelet.parameters(), lr=lr)

        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(max_epochs):
            optimizer.zero_grad()

            # 前向传播
            class_signals = class_signals.squeeze()
            coeffs = self.wavelet(class_signals)
            reconstructed = self.wavelet.idwt_multilevel(coeffs, self.signal_length)

            # 重构损失
            loss = F.mse_loss(reconstructed, class_signals)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 早停
            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

            if verbose and epoch % 20 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

        # 冻结参数
        for param in self.wavelet.parameters():
            param.requires_grad = False
        self.wavelet.eval()

        self.best_loss = best_loss

        if verbose:
            print(f"  Final loss: {best_loss:.6f}")

        return best_loss

    def compute_reconstruction_error(self, signals, device='cpu'):
        """计算重构误差

        Args:
            signals: (N, signal_length) 测试信号

        Returns:
            errors: (N,) 每个信号的重构误差
        """
        self.wavelet = self.wavelet.to(device)
        self.wavelet.eval()

        with torch.no_grad():
            coeffs = self.wavelet(signals)
            reconstructed = self.wavelet.idwt_multilevel(coeffs, self.signal_length)

            # 计算每个样本的误差
            errors = (signals - reconstructed).pow(2).mean(dim=1)

        return errors.cpu().numpy()


class ReconstructionBasedClassifier:
    """基于重构误差的分类器

    训练: 每个类别训练一个小波
    预测: 用重构误差最小的类别
    """

    def __init__(self, filter_length, levels, signal_length, num_classes):
        """
        Args:
            filter_length: 滤波器长度
            levels: DWT层数
            signal_length: 信号长度
            num_classes: 类别数
        """
        self.filter_length = filter_length
        self.levels = levels
        self.signal_length = signal_length
        self.num_classes = num_classes

        # 为每个类别创建一个小波重构器
        self.reconstructors = [
            SingleWaveletReconstructor(filter_length, levels, signal_length)
            for _ in range(num_classes)
        ]

        self.trained = False

    def train(self, train_loader, target_loss=0.01, max_epochs=500, device='cpu', verbose=True):
        """训练所有类别的小波到相同的loss水平

        Args:
            train_loader: PyTorch DataLoader
            target_loss: 目标loss (所有类别都训练到这个水平)
            max_epochs: 最大训练轮数
            device: 'cpu' or 'cuda'
            verbose: 是否打印进度
        """
        if verbose:
            print("=" * 70)
            print("训练基于重构的分类器")
            print(f"目标: 所有类别训练到 loss = {target_loss}")
            print("=" * 70)

        # 按类别分组数据
        signals_by_class = [[] for _ in range(self.num_classes)]

        for signals, labels in train_loader:
            for class_i in range(self.num_classes):
                mask = labels == class_i
                if mask.any():
                    signals_by_class[class_i].append(signals[mask])

        # 合并每个类别的数据
        for class_i in range(self.num_classes):
            signals_by_class[class_i] = torch.cat(signals_by_class[class_i], dim=0)

        if verbose:
            for class_i in range(self.num_classes):
                print(f"类别 {class_i}: {len(signals_by_class[class_i])} 个样本")

        # 训练每个类别的小波到目标loss
        final_losses = []
        for class_i in range(self.num_classes):
            if verbose:
                print(f"\n训练类别 {class_i} 的小波到 loss={target_loss}...")

            class_signals = signals_by_class[class_i].to(device)
            class_signals = class_signals.squeeze()
            loss = self.reconstructors[class_i].train_to_target_loss(
                class_signals,
                target_loss=target_loss,
                max_epochs=max_epochs,
                device=device,
                verbose=verbose
            )

            final_losses.append(loss)

            if verbose:
                print(f"✓ 类别 {class_i}: 最终 loss = {loss:.6f}")

        if verbose:
            print("\n" + "=" * 70)
            print("训练完成!")
            print(f"Loss 范围: [{min(final_losses):.6f}, {max(final_losses):.6f}]")
            print(f"Loss 标准差: {np.std(final_losses):.6f}")
            print("=" * 70)

        self.trained = True

    def predict(self, test_loader, device='cpu', verbose=True):
        """预测

        对每个测试样本:
          1. 用每个类别的小波重构
          2. 计算重构误差
          3. 选择误差最小的类别

        Returns:
            predictions: 预测类别
            true_labels: 真实类别 (如果有)
            errors_matrix: 误差矩阵
        """
        if not self.trained:
            raise RuntimeError("模型未训练! 先调用 train()")

        all_predictions = []
        all_labels = []
        all_errors = []

        for batch in test_loader:
            if len(batch) == 2:
                signals, labels = batch
                all_labels.append(labels.numpy())
            else:
                signals = batch

            signals = signals.to(device)
            signals = signals.squeeze()
            batch_size = len(signals)

            # 计算每个类别的重构误差
            errors_per_class = np.zeros((batch_size, self.num_classes))

            for class_i in range(self.num_classes):
                errors = self.reconstructors[class_i].compute_reconstruction_error(
                    signals, device
                )
                errors_per_class[:, class_i] = errors

            # 选择误差最小的类别
            predictions = np.argmin(errors_per_class, axis=1)

            all_predictions.append(predictions)
            all_errors.append(errors_per_class)

        predictions = np.concatenate(all_predictions)
        errors_matrix = np.vstack(all_errors)

        if verbose:
            print("\n重构误差统计:")
            print("-" * 50)
            for class_i in range(self.num_classes):
                avg_error = errors_matrix[:, class_i].mean()
                std_error = errors_matrix[:, class_i].std()
                print(f"类别 {class_i}: {avg_error:.6f} ± {std_error:.6f}")

        if all_labels:
            true_labels = np.concatenate(all_labels)
            return predictions, true_labels, errors_matrix
        else:
            return predictions, None, errors_matrix

    def evaluate(self, test_loader, device='cpu'):
        """评估模型

        Returns:
            accuracy: 准确率
            report: 分类报告
            conf_matrix: 混淆矩阵
            errors_matrix: 误差矩阵
        """
        predictions, true_labels, errors_matrix = self.predict(
            test_loader, device, verbose=False
        )

        if true_labels is None:
            raise ValueError("测试集没有标签!")

        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)

        return accuracy, report, conf_matrix, errors_matrix


# ============================================================================
# 使用示例
# ============================================================================

class WaveletCoefficientsClassifier:
    """用多个类别特定小波的系数做分类"""

    def __init__(self, filter_length, levels, signal_length, num_classes,
                 n_estimators=200):
        """
        Args:
            filter_length: 滤波器长度
            levels: DWT层数
            signal_length: 信号长度
            num_classes: 类别数
            n_estimators: ExtraTreesClassifier的树数量
        """
        self.filter_length = filter_length
        self.levels = levels
        self.signal_length = signal_length
        self.num_classes = num_classes

        # 为每个类别创建一个小波
        self.reconstructors = [
            SingleWaveletReconstructor(filter_length, levels, signal_length)
            for _ in range(num_classes)
        ]

        # ExtraTreesClassifier
        self.classifier = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

        self.trained_wavelets = False
        self.trained_classifier = False

    def train_wavelets(self, train_loader, target_loss=0.01, max_epochs=500,
                       device='cpu', verbose=True):
        """训练所有类别的小波到相同loss水平

        Args:
            train_loader: PyTorch DataLoader
            target_loss: 目标loss
            max_epochs: 最大轮数
            device: 'cpu' or 'cuda'
            verbose: 是否打印
        """
        if verbose:
            print("=" * 70)
            print("阶段1: 训练每个类别的小波")
            print(f"目标: 所有类别训练到 loss = {target_loss}")
            print("=" * 70)

        # 按类别分组
        signals_by_class = [[] for _ in range(self.num_classes)]

        for signals, labels in train_loader:
            for class_i in range(self.num_classes):
                mask = labels == class_i
                if mask.any():
                    signals_by_class[class_i].append(signals[mask])

        # 合并
        for class_i in range(self.num_classes):
            signals_by_class[class_i] = torch.cat(signals_by_class[class_i], dim=0)

        if verbose:
            for class_i in range(self.num_classes):
                print(f"类别 {class_i}: {len(signals_by_class[class_i])} 个样本")

        # 训练每个小波
        final_losses = []
        for class_i in range(self.num_classes):
            if verbose:
                print(f"\n训练类别 {class_i} 的小波...")

            class_signals = signals_by_class[class_i].to(device)

            loss = self.reconstructors[class_i].train_to_target_loss(
                class_signals,
                target_loss=target_loss,
                max_epochs=max_epochs,
                device=device,
                verbose=verbose
            )

            final_losses.append(loss)

            if verbose:
                print(f"✓ 类别 {class_i}: 最终 loss = {loss:.6f}")

        if verbose:
            print("\n" + "=" * 70)
            print("所有小波训练完成!")
            print(f"Loss 范围: [{min(final_losses):.6f}, {max(final_losses):.6f}]")
            print(f"Loss 标准差: {np.std(final_losses):.6f}")
            print("=" * 70)

        self.trained_wavelets = True

    def extract_all_coefficients(self, signals, device='cpu'):
        """用所有小波提取系数并拼接

        Args:
            signals: (N, signal_length) 信号
            device: 'cpu' or 'cuda'

        Returns:
            features: (N, total_feature_dim) 所有小波系数拼接
        """
        signals = signals.to(device)
        signals = signals.squeeze()
        all_features = []

        for class_i in range(self.num_classes):
            wavelet = self.reconstructors[class_i].wavelet.to(device)
            wavelet.eval()

            with torch.no_grad():
                # 提取小波系数
                coeffs = wavelet(signals)

                # 展平并拼接
                flat_coeffs = torch.cat([c.flatten(1) for c in coeffs], dim=1)

                all_features.append(flat_coeffs.cpu().numpy())

        # 拼接所有类别的小波系数
        combined_features = np.hstack(all_features)

        return combined_features

    def train_classifier(self, train_loader, device='cpu', verbose=True):
        """训练ExtraTreesClassifier

        Args:
            train_loader: PyTorch DataLoader
            device: 'cpu' or 'cuda'
            verbose: 是否打印
        """
        if not self.trained_wavelets:
            raise RuntimeError("小波未训练! 先调用 train_wavelets()")

        if verbose:
            print("\n" + "=" * 70)
            print("阶段2: 训练ExtraTreesClassifier")
            print("=" * 70)

        # 提取所有训练数据的特征
        all_features = []
        all_labels = []

        for signals, labels in train_loader:
            features = self.extract_all_coefficients(signals, device)
            all_features.append(features)
            all_labels.append(labels.numpy())

        X_train = np.vstack(all_features)
        y_train = np.concatenate(all_labels)

        if verbose:
            print(f"特征维度: {X_train.shape}")
            print(f"  每个类别的小波提取: {X_train.shape[1] // self.num_classes} 维")
            print(f"  总共 {self.num_classes} 个小波: {X_train.shape[1]} 维")

        # 训练分类器
        if verbose:
            print("\n训练ExtraTreesClassifier...")

        self.classifier.fit(X_train, y_train)

        train_acc = self.classifier.score(X_train, y_train)

        if verbose:
            print(f"训练集准确率: {train_acc * 100:.2f}%")
            print("✓ 训练完成")

        self.trained_classifier = True

    def predict(self, test_loader, device='cpu', verbose=True):
        """预测

        Returns:
            predictions: 预测类别
            true_labels: 真实类别
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

            # 提取特征
            features = self.extract_all_coefficients(signals, device)

            # 预测
            preds = self.classifier.predict(features)
            all_predictions.append(preds)

        predictions = np.concatenate(all_predictions)

        if all_labels:
            true_labels = np.concatenate(all_labels)
            return predictions, true_labels
        else:
            return predictions, None

    def evaluate(self, test_loader, device='cpu'):
        """评估"""
        predictions, true_labels = self.predict(test_loader, device, verbose=False)

        if true_labels is None:
            raise ValueError("测试集没有标签!")

        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)

        return accuracy, report, conf_matrix

    def train(self, train_loader, target_loss=0.01, max_epochs=500,
              device='cpu', verbose=True):
        """完整训练流程

        Args:
            train_loader: PyTorch DataLoader
            target_loss: 小波训练目标loss
            max_epochs: 小波训练最大轮数
            device: 'cpu' or 'cuda'
            verbose: 是否打印
        """
        # 阶段1: 训练小波
        self.train_wavelets(train_loader, target_loss, max_epochs, device, verbose)

        # 阶段2: 训练分类器
        self.train_classifier(train_loader, device, verbose)


# ============================================================================
# 主程序示例
# ============================================================================

if __name__ == "__main__":
    from utils import create_dataloader_from_arff


    # 创建模型
    model = WaveletCoefficientsClassifier(
        filter_length=8,
        levels=3,
        signal_length=720,  # 你的信号长度
        num_classes=2  # 你的类别数
    )



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


    # 训练 (每个类别训练一个小波)
    model.train(train_dataloader, max_epochs=100000, target_loss=0.05, verbose=True)

    # 预测
    predictions, true_labels = model.predict(test_dataloader, device='cpu')

    # 评估
    accuracy, report, conf_matrix= model.evaluate(test_dataloader)
    print(f"准确率: {accuracy * 100:.2f}%")
    print(report)
