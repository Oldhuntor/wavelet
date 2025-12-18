"""
Simple Wavelet Classifier - 最简化版本
====================================

No switches, just works.
固定配置: Wavelet + PCA + Stats + FFT
"""

import torch
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
import multiprocessing as mp

try:
    from model.adaptive_filters import LearnableWaveletTransform
except ImportError:
    print("ERROR: Need wavelet_classifier.py")
    raise


class SingleWaveletModel:
    """Single wavelet model"""

    def __init__(self, filter_length, levels, signal_length):
        self.wavelet = LearnableWaveletTransform(
            filter_length, levels, 'random',
            use_frequency_constraint=False,
            use_learnable_activation=True
        )
        self.signal_length = signal_length
        self.filter_length = filter_length
        self.levels = levels

    def extract_features(self, signals, device='cpu'):
        """Extract wavelet features"""
        self.wavelet = self.wavelet.to(device)
        self.wavelet.eval()

        with torch.no_grad():
            coeffs = self.wavelet(signals)
            flat_coeffs = torch.cat([c.flatten(1) for c in coeffs], dim=1)

        return flat_coeffs.cpu().numpy()

    def train_filters(self, train_data, max_epochs=1, device='cpu'):
        """Train wavelet filters"""
        import torch.nn.functional as F

        self.wavelet = self.wavelet.to(device)
        self.wavelet.train()

        optimizer = torch.optim.Adam(self.wavelet.parameters(), lr=0.0001)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for signals, _ in train_data:
                signals = signals.to(device)
                signals.squeeze_()
                optimizer.zero_grad()

                coeffs = self.wavelet(signals)
                reconstructed = self.wavelet.idwt_multilevel(coeffs, self.signal_length)
                loss = F.mse_loss(reconstructed, signals)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 10:
                break

        # Freeze
        for param in self.wavelet.parameters():
            param.requires_grad = False
        self.wavelet.eval()

        return best_loss


def train_single_model_worker(args):
    """Worker to train one model"""
    model_idx, filter_length, levels, signal_length, train_data, device = args

    print(f"  Worker {model_idx}: Training (filter_length={filter_length})")

    model = SingleWaveletModel(filter_length, levels, signal_length)
    best_loss = model.train_filters(train_data, max_epochs=100, device=device)

    print(f"  Worker {model_idx}: Done (loss={best_loss:.6f})")

    return model_idx, model, best_loss


def extract_features_worker(args):
    """Worker to extract features"""
    model_idx, model, signals, device = args
    features = model.extract_features(signals, device)
    return model_idx, features

# 1.35 0.69;;; 2.86 0.60;; 0.314 0.768 ;; 0.05 0.552 ;; 0.185  0.68
class SimpleWaveletClassifier:
    """
    Simple Wavelet Classifier

    固定配置:
      - 多个wavelet models (并行训练)
      - PCA降维
      - 统计特征 (mean, std, max, min)
      - FFT特征
      - ExtraTreesClassifier

    Usage:
        model = SimpleWaveletClassifier(
            filter_lengths=[4, 8, 16, 32],
            levels=3,
            signal_length=256,
            num_classes=3
        )
        model.train(train_loader)
        preds, labels = model.predict(test_loader)
    """

    def __init__(self, filter_lengths, levels, signal_length, num_classes,
                 n_estimators=500, pca_variance=0.95, n_jobs=None):
        """
        Args:
            filter_lengths: List of filter lengths [4, 8, 16, 32]
            levels: DWT levels (3 or 4)
            signal_length: Input signal length
            num_classes: Number of classes
            n_estimators: Number of trees
            pca_variance: PCA variance to keep (0.95 = 95%)
            n_jobs: CPU cores to use (None = all)
        """
        self.filter_lengths = filter_lengths if isinstance(filter_lengths, list) else [filter_lengths]
        self.num_models = len(self.filter_lengths)
        self.levels = levels
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.n_jobs = n_jobs or mp.cpu_count()
        self.pca_variance = pca_variance

        self.models = None
        self.pca = None
        self.classifier = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        self.trained = False

    def _compute_feature_sizes(self):
        """Compute feature size for each model"""
        self.model_feature_sizes = []
        dummy_signal = torch.zeros(1, self.signal_length)

        for model in self.models:
            coeffs = model.wavelet(dummy_signal)
            feature_size = sum(c.numel() for c in coeffs)
            self.model_feature_sizes.append(feature_size)

    def _train_models(self, train_loader, device='cpu'):
        """Train all wavelet models in parallel"""
        print("\n" + "=" * 70)
        print(f"Stage 1: Training {self.num_models} wavelet models")
        print(f"Using {self.n_jobs} parallel workers")
        print("=" * 70)

        # Convert to list for pickling
        train_data = [(signals, labels) for signals, labels in train_loader]

        # Prepare worker arguments
        worker_args = [
            (i, self.filter_lengths[i], self.levels, self.signal_length, train_data, device)
            for i in range(self.num_models)
        ]

        # Train in parallel
        with mp.Pool(self.n_jobs) as pool:
            results = pool.map(train_single_model_worker, worker_args)

        # Sort and extract
        results.sort(key=lambda x: x[0])
        self.models = [model for _, model, _ in results]
        losses = [loss for _, _, loss in results]

        # Compute feature sizes
        self._compute_feature_sizes()

        print("\n" + "=" * 70)
        print("Training Results:")
        for i, loss in enumerate(losses):
            print(f"  Model {i} (len={self.filter_lengths[i]}): loss={loss:.6f}")
        print(f"  Average loss: {np.mean(losses):.6f}")
        print("=" * 70)

    def _extract_all_features(self, signals, device='cpu'):
        """Extract wavelet features from all models"""
        # Prepare workers
        worker_args = [
            (i, self.models[i], signals, device)
            for i in range(self.num_models)
        ]

        # Extract in parallel
        with mp.Pool(self.n_jobs) as pool:
            results = pool.map(extract_features_worker, worker_args)

        # Sort and concatenate
        results.sort(key=lambda x: x[0])
        all_features = [features for _, features in results]

        return np.hstack(all_features)

    def _add_stats_and_fft(self, wavelet_features, raw_signals):
        """Add stats and FFT features"""
        features = wavelet_features.copy()

        # Add stats (per model)
        start_idx = 0
        for feature_size in self.model_feature_sizes:
            end_idx = start_idx + feature_size
            model_feats = wavelet_features[:, start_idx:end_idx]

            stats = np.column_stack([
                model_feats.mean(axis=1),
                model_feats.std(axis=1),
                model_feats.max(axis=1),
                model_feats.min(axis=1)
            ])
            features = np.hstack([features, stats])
            start_idx = end_idx

        # Add FFT
        signals_clean = np.nan_to_num(raw_signals, nan=0.0, posinf=0.0, neginf=0.0)
        signals_mean = signals_clean.mean(axis=1, keepdims=True)
        signals_std = signals_clean.std(axis=1, keepdims=True)
        signals_std[signals_std < 1e-10] = 1.0
        signals_normalized = (signals_clean - signals_mean) / signals_std

        fft_features = np.abs(np.fft.rfft(signals_normalized, axis=1))
        fft_features = np.nan_to_num(fft_features, nan=0.0, posinf=0.0, neginf=0.0)
        fft_features = np.clip(fft_features, -1e10, 1e10)

        features = np.hstack([features, fft_features])

        return features

    def train(self, train_loader, device='cpu'):
        """
        Complete training pipeline

        Args:
            train_loader: PyTorch DataLoader
            device: 'cpu' or 'cuda'
        """
        # Stage 1: Train wavelet models
        self._train_models(train_loader, device)

        # Stage 2: Extract features and train classifier
        print("\n" + "=" * 70)
        print("Stage 2: Training classifier")
        print("=" * 70)

        all_wavelet_features = []
        all_signals = []
        all_labels = []

        for signals, labels in train_loader:
            signals_cpu = signals.to(device)
            signals.squeeze_()

            # Extract wavelet features
            wavelet_feats = self._extract_all_features(signals_cpu, device)
            all_wavelet_features.append(wavelet_feats)
            all_signals.append(signals.cpu().numpy())
            all_labels.append(labels.numpy())

        X_wavelet = np.vstack(all_wavelet_features)
        signals_np = np.vstack(all_signals)
        y_train = np.concatenate(all_labels)

        print(f"Wavelet features: {X_wavelet.shape}")

        # Apply PCA
        self.pca = PCA(n_components=self.pca_variance)
        X_reduced = self.pca.fit_transform(X_wavelet)
        print(f"After PCA: {X_reduced.shape} (variance={self.pca.explained_variance_ratio_.sum():.3f})")

        # Add stats and FFT
        X_final = self._add_stats_and_fft(X_wavelet, signals_np)

        # Replace wavelet part with PCA version
        n_pca = X_reduced.shape[1]
        n_extra = X_final.shape[1] - X_wavelet.shape[1]
        X_final = np.hstack([X_reduced, X_final[:, -n_extra:]])

        print(f"Final features: {X_final.shape}")

        # Train classifier
        self.classifier.fit(X_final, y_train)
        train_acc = self.classifier.score(X_final, y_train)

        print(f"Training accuracy: {train_acc * 100:.2f}%")
        print("✓ Training complete")

        self.trained = True

    def predict(self, test_loader, device='cpu'):
        """
        Predict on test data

        Args:
            test_loader: PyTorch DataLoader
            device: 'cpu' or 'cuda'

        Returns:
            predictions, true_labels (if available)
        """
        if not self.trained:
            raise RuntimeError("Model not trained! Call train() first.")

        all_preds = []
        all_labels = []

        for batch in test_loader:
            if len(batch) == 2:
                signals, labels = batch
                all_labels.append(labels.numpy())
            else:
                signals = batch

            signals_cpu = signals.to(device)
            signals.squeeze_()
            signals_np = signals.cpu().numpy()

            # Extract wavelet features
            wavelet_feats = self._extract_all_features(signals_cpu, device)

            # Apply PCA
            X_reduced = self.pca.transform(wavelet_feats)

            # Add stats and FFT
            X_final = self._add_stats_and_fft(wavelet_feats, signals_np)

            # Replace wavelet with PCA
            n_pca = X_reduced.shape[1]
            n_extra = X_final.shape[1] - wavelet_feats.shape[1]
            X_final = np.hstack([X_reduced, X_final[:, -n_extra:]])

            # Predict
            preds = self.classifier.predict(X_final)
            all_preds.append(preds)

        predictions = np.concatenate(all_preds)

        if all_labels:
            true_labels = np.concatenate(all_labels)
            return predictions, true_labels
        else:
            return predictions


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    from utils import create_dataloader_from_arff

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
    # 创建
    model = SimpleWaveletClassifier(
        filter_lengths=[3, 16, 16, 40, 40, 40],
        levels=3,
        signal_length=720,
        num_classes=2
    )

    # 训练
    model.train(train_dataloader)

    # 预测
    preds, labels = model.predict(test_dataloader)

    # 评估
    from sklearn.metrics import accuracy_score, confusion_matrix
    cm = confusion_matrix(labels, preds)
    print(cm)
    acc = accuracy_score(labels, preds)
    print(f"Accuracy: {acc:.3f}")