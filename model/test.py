import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import multiprocessing as mp
from functools import partial
import pickle

# Import the single-pair wavelet transform
# This is from wavelet_classifier.py - the basic one with just low_pass + high_pass
try:
    from model.adaptive_filters import LearnableWaveletTransform
except ImportError:
    print("ERROR: Need wavelet_classifier.py with LearnableWaveletTransform")
    print("This uses the SINGLE PAIR version (low_pass + high_pass)")
    raise


class SingleWaveletModel:
    """Single wavelet transform model (can be pickled)

    Uses LearnableWaveletTransform from wavelet_classifier.py
    Each model has ONE pair of filters: (low_pass, high_pass)
    """

    def __init__(self, filter_length, levels, signal_length,
                 use_frequency_constraint=False, use_learnable_activation=False):
        # This creates ONE pair of filters
        self.wavelet = LearnableWaveletTransform(
            filter_length, levels, 'random',
            use_frequency_constraint, use_learnable_activation
        )
        self.signal_length = signal_length
        self.filter_length = filter_length
        self.levels = levels

    def extract_features(self, signals, device='cpu'):
        """Extract features from signals"""
        self.wavelet = self.wavelet.to(device)
        self.wavelet.eval()

        with torch.no_grad():
            coeffs = self.wavelet(signals)
            flat_coeffs = torch.cat([c.flatten(1) for c in coeffs], dim=1)

        return flat_coeffs.cpu().numpy()

    def train_filters(self, train_data, max_epochs=100, initial_lr=0.001,
                      min_delta=1e-6, patience=10, device='cpu'):
        """Train wavelet filters"""
        self.wavelet = self.wavelet.to(device)
        self.wavelet.train()

        optimizer = torch.optim.Adam(self.wavelet.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )

        best_loss = float('inf')
        epochs_no_improve = 0

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
            scheduler.step(avg_loss)

            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        # Freeze
        for param in self.wavelet.parameters():
            param.requires_grad = False
        self.wavelet.eval()

        return best_loss


# ============================================================================
# Multiprocessing worker functions
# ============================================================================

def train_single_model_worker(args):
    """Worker function to train one model

    Args:
        args: (model_idx, filter_length, levels, signal_length, train_data, device,
               max_epochs, initial_lr, patience)

    Returns:
        (model_idx, trained_model, best_loss)
    """
    model_idx, filter_length, levels, signal_length, train_data, device, max_epochs, initial_lr, patience = args

    print(f"  Worker {model_idx}: Starting training (filter_length={filter_length})")

    # Create and train model
    model = SingleWaveletModel(filter_length, levels, signal_length)
    best_loss = model.train_filters(train_data,
                                    max_epochs=max_epochs,
                                    initial_lr=initial_lr,
                                    patience=patience,
                                    device=device)

    print(f"  Worker {model_idx}: Finished (loss={best_loss:.6f})")

    return model_idx, model, best_loss


def extract_features_worker(args):
    """Worker function to extract features from one model

    Args:
        args: (model_idx, model, signals, device)

    Returns:
        (model_idx, features)
    """
    model_idx, model, signals, device = args
    features = model.extract_features(signals, device)
    return model_idx, features


# ============================================================================
# Main multi-model classifier
# ============================================================================

class MultiModelParallelClassifier:
    """Multiple independent models trained in parallel

    Architecture:
        N models → Extract features in parallel → Concat → Classifier

    Uses multiprocessing for:
        - Parallel training of N models
        - Parallel feature extraction
    """

    def __init__(self, filter_lengths, levels, signal_length, num_classes,
                 n_estimators=200, max_depth=None, random_state=42,
                 n_jobs=None,
                 use_wavelet=True, use_pca=True, pca_components=0.95,
                 use_stats=True, use_fft=True,
                 max_epochs=100, initial_lr=0.001, patience=10):
        """
        Args:
            filter_lengths: List of filter lengths, one per model
            levels: DWT levels
            signal_length: Input signal length
            num_classes: Number of classes
            n_estimators: ExtraTrees param
            max_depth: ExtraTrees param
            random_state: Random seed
            n_jobs: Number of parallel workers (None = all CPUs)

            # Feature options (GLOBAL SWITCHES)
            use_wavelet: Use wavelet transform features (if False, only stats/fft on raw signal)
            use_pca: Use PCA to reduce wavelet coefficients dimensionality
            pca_components: PCA components (float: variance ratio, int: n_components)
            use_stats: Add statistical features (mean, std, max, min)
            use_fft: Add FFT features

            # Training options
            max_epochs: Maximum epochs for training wavelet filters
            initial_lr: Initial learning rate for filter training
            patience: Early stopping patience
        """
        self.filter_lengths = filter_lengths if isinstance(filter_lengths, list) else [filter_lengths]
        self.num_models = len(self.filter_lengths)
        self.levels = levels
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.n_jobs = 10

        # Feature extraction options (GLOBAL SWITCHES)
        self.use_wavelet = use_wavelet
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.use_stats = use_stats
        self.use_fft = use_fft
        self.pca = None

        # Training options
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr
        self.patience = patience

        # Models (will be trained only if use_wavelet=True)
        self.models = None

        # Classifier
        self.classifier = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.classifier_fitted = False

    def train_models_parallel(self, train_loader, device='cpu', verbose=True):
        """Train all models in parallel using multiprocessing

        Stage 1: Parallel training of N independent models
        Uses max_epochs, initial_lr, and patience from __init__

        If use_wavelet=False, this step is skipped
        """
        if not self.use_wavelet:
            if verbose:
                print("\n" + "=" * 70)
                print("Stage 1: SKIPPED (use_wavelet=False)")
                print("Will use only stats/FFT features on raw signal")
                print("=" * 70)
            return []

        print("\n" + "=" * 70)
        print(f"Stage 1: Training {self.num_models} models in parallel")
        print(f"Using {self.n_jobs} workers")
        print(f"Training config: max_epochs={self.max_epochs}, lr={self.initial_lr}, patience={self.patience}")
        print("=" * 70)

        # Convert DataLoader to list (for pickling)
        train_data = [(signals, labels) for signals, labels in train_loader]

        # Prepare arguments for each worker (now includes training params)
        worker_args = [
            (i, self.filter_lengths[i], self.levels, self.signal_length,
             train_data, device, self.max_epochs, self.initial_lr, self.patience)
            for i in range(self.num_models)
        ]

        # Train in parallel
        with mp.Pool(self.n_jobs) as pool:
            results = pool.map(train_single_model_worker, worker_args)

        # Sort results by model_idx
        results.sort(key=lambda x: x[0])

        # Extract models
        self.models = [model for _, model, _ in results]
        losses = [loss for _, _, loss in results]

        if verbose:
            print("\n" + "=" * 70)
            print("Training Results:")
            for i, loss in enumerate(losses):
                print(f"  Model {i} (len={self.filter_lengths[i]}): loss={loss:.6f}")
            print(f"  Average loss: {np.mean(losses):.6f}")
            print("=" * 70)

        return losses

    def extract_features_parallel(self, signals, device='cpu',
                                  add_stats=True, add_fft=True):
        """Extract features from all models in parallel

        Args:
            signals: Input signals
            device: 'cpu' or 'cuda'
            add_stats: Add statistical features (mean, std, max, min)
            add_fft: Add FFT features

        Returns:
            Combined features with optional stats and FFT
        """
        if self.models is None:
            raise RuntimeError("Models not trained yet!")

        # Prepare arguments for each worker
        worker_args = [
            (i, self.models[i], signals, device)
            for i in range(self.num_models)
        ]

        # Extract wavelet features in parallel
        with mp.Pool(self.n_jobs) as pool:
            results = pool.map(extract_features_worker, worker_args)

        # Sort by model_idx and concatenate
        results.sort(key=lambda x: x[0])
        all_features = [features for _, features in results]

        # Concatenate wavelet features from all models
        wavelet_features = np.hstack(all_features)

        # Optionally add statistical features
        if add_stats:
            stats_features = []
            for features in all_features:
                # Per-model statistics
                stats = np.column_stack([
                    features.mean(axis=1),  # Mean
                    features.std(axis=1),  # Std
                    features.max(axis=1),  # Max
                    features.min(axis=1)  # Min
                ])
                stats_features.append(stats)

            stats_combined = np.hstack(stats_features)
            wavelet_features = np.hstack([wavelet_features, stats_combined])

        # Optionally add FFT features
        if add_fft:
            # Convert signals to numpy if needed
            if torch.is_tensor(signals):
                signals_np = signals.cpu().numpy()
            else:
                signals_np = np.array(signals)

            # Clean signals: replace inf/nan with 0
            signals_np = np.nan_to_num(signals_np, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize per sample (防止数值过大)
            signals_mean = signals_np.mean(axis=1, keepdims=True)
            signals_std = signals_np.std(axis=1, keepdims=True)
            signals_std[signals_std < 1e-10] = 1.0  # Avoid division by zero
            signals_normalized = (signals_np - signals_mean) / signals_std

            # Compute FFT
            fft_features = np.abs(np.fft.rfft(signals_normalized, axis=1))

            # Clean FFT result: replace inf/nan and clip
            fft_features = np.nan_to_num(fft_features, nan=0.0, posinf=0.0, neginf=0.0)
            fft_features = np.clip(fft_features, -1e10, 1e10)

            wavelet_features = np.hstack([wavelet_features, fft_features])

        return wavelet_features

    def train_classifier(self, train_loader, device='cpu', verbose=True):
        """Train classifier on combined features with PCA

        Pipeline (uses global switches from __init__):
          1. Extract wavelet coefficients (if use_wavelet=True)
          2. Apply PCA (if use_pca=True and use_wavelet=True)
          3. Add statistical features (if use_stats=True)
          4. Add FFT features (if use_fft=True)
          5. Train classifier

        Args:
            train_loader: DataLoader
            device: 'cpu' or 'cuda'
            verbose: Print progress
        """
        from sklearn.decomposition import PCA

        print("\n" + "=" * 70)
        print("Stage 2: Training classifier on combined features")
        print(
            f"Feature switches: Wavelet={self.use_wavelet}, PCA={self.use_pca}, Stats={self.use_stats}, FFT={self.use_fft}")
        print("=" * 70)

        all_signals = []
        all_labels = []
        all_wavelet_features = [] if self.use_wavelet else None

        # Extract features from all batches
        for signals, labels in train_loader:
            all_signals.append(signals.cpu().numpy())
            all_labels.append(labels.numpy())

            if self.use_wavelet:
                signals_cpu = signals.to(device)
                # signals.squeeze_()
                signals_cpu.squeeze_()
                # Extract wavelet features (parallel, no stats/fft yet)
                features = self.extract_features_parallel(signals_cpu, device,
                                                          add_stats=False, add_fft=False)
                all_wavelet_features.append(features)

        signals_np = np.vstack(all_signals)
        y_train = np.concatenate(all_labels)

        # Start with wavelet features or empty
        if self.use_wavelet:
            X_wavelet = np.vstack(all_wavelet_features)

            if verbose:
                print(f"Original wavelet features: {X_wavelet.shape}")

            # Apply PCA to wavelet coefficients
            if self.use_pca:
                self.pca = PCA(n_components=self.pca_components)
                X_features = self.pca.fit_transform(X_wavelet)

                if verbose:
                    if isinstance(self.pca_components, float):
                        print(f"PCA (variance={self.pca_components}): {X_wavelet.shape[1]} → {X_features.shape[1]}")
                        print(f"  Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
                    else:
                        print(f"PCA (n_components={self.pca_components}): {X_wavelet.shape[1]} → {X_features.shape[1]}")
            else:
                X_features = X_wavelet
                if verbose:
                    print("PCA disabled, using all wavelet features")
        else:
            # No wavelet features - start with empty
            X_features = np.empty((len(signals_np), 0))
            if verbose:
                print("Wavelet features disabled, using only stats/FFT")

        # Add statistical features
        if self.use_stats:
            if self.use_wavelet:
                # Stats on wavelet coefficients per model

                X_wavelet = np.vstack(all_wavelet_features)
                model_features = np.array_split(X_wavelet, self.num_models, axis=1)
                stats_features = []
                for features in model_features:
                    stats = np.column_stack([
                        features.mean(axis=1),
                        features.std(axis=1),
                        features.max(axis=1),
                        features.min(axis=1)
                    ])
                    stats_features.append(stats)

                stats_combined = np.hstack(stats_features)
            else:
                # Stats on raw signal
                stats_combined = np.column_stack([
                    signals_np.mean(axis=1),
                    signals_np.std(axis=1),
                    signals_np.max(axis=1),
                    signals_np.min(axis=1)
                ])

            X_features = np.hstack([X_features, stats_combined])

            if verbose:
                print(f"  + Statistical features: {stats_combined.shape[1]}")

        # Add FFT features
        if self.use_fft:
            # Clean signals
            signals_clean = np.nan_to_num(signals_np, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize
            signals_mean = signals_clean.mean(axis=1, keepdims=True)
            signals_std = signals_clean.std(axis=1, keepdims=True)
            signals_std[signals_std < 1e-10] = 1.0
            signals_normalized = (signals_clean - signals_mean) / signals_std

            # Compute FFT
            fft_features = np.abs(np.fft.rfft(signals_normalized, axis=1))

            # Clean FFT result
            fft_features = np.nan_to_num(fft_features, nan=0.0, posinf=0.0, neginf=0.0)
            fft_features = np.clip(fft_features, -1e10, 1e10)

            if not self.use_wavelet:
                X_features = fft_features.squeeze()
            else:
                fft_features = fft_features.squeeze()
                X_features = np.hstack([X_features, fft_features])

            if verbose:
                print(f"  + FFT features: {fft_features.shape[1]}")

        if verbose:
            print(f"Final features: {X_features.shape}")

        # Check if we have any features
        if X_features.shape[1] == 0:
            raise ValueError("No features enabled! Set at least one of use_wavelet, use_stats, or use_fft to True")

        # Train classifier
        self.classifier.fit(X_features, y_train)
        self.classifier_fitted = True

        if verbose:
            train_acc = self.classifier.score(X_features, y_train)
            print(f"Training accuracy: {train_acc * 100:.2f}%")

        print("✓ Classifier trained")

    def predict(self, test_loader, device='cpu'):
        """Predict on test data

        Applies same transformation pipeline as training (using global switches)
        """
        if not self.classifier_fitted:
            raise RuntimeError("Classifier not trained yet!")

        all_preds = []
        all_labels = []

        for batch in test_loader:
            if len(batch) == 2:
                signals, labels = batch
                all_labels.append(labels.numpy())
            else:
                signals = batch

            signals_cpu = signals.to(device)
            signals_np = signals.cpu().numpy()

            # Start with empty or wavelet features
            if self.use_wavelet:
                # Extract wavelet features (without stats/fft)
                signals_cpu = signals_cpu.squeeze()
                wavelet_features = self.extract_features_parallel(signals_cpu, device,
                                                                  add_stats=False, add_fft=False)

                # Apply PCA if used
                if self.use_pca and self.pca is not None:
                    features = self.pca.transform(wavelet_features)
                else:
                    features = wavelet_features
            else:
                # No wavelet features
                features = np.empty((len(signals_np), 0))
                wavelet_features = None

            # Add stats if used
            if self.use_stats:
                if self.use_wavelet and wavelet_features is not None:
                    # Stats on wavelet coefficients
                    model_features = np.array_split(wavelet_features, self.num_models, axis=1)
                    stats_features = []
                    for mf in model_features:
                        stats = np.column_stack([
                            mf.mean(axis=1),
                            mf.std(axis=1),
                            mf.max(axis=1),
                            mf.min(axis=1)
                        ])
                        stats_features.append(stats)
                    stats_combined = np.hstack(stats_features)
                else:
                    # Stats on raw signal
                    stats_combined = np.column_stack([
                        signals_np.mean(axis=1),
                        signals_np.std(axis=1),
                        signals_np.max(axis=1),
                        signals_np.min(axis=1)
                    ])

                features = np.hstack([features, stats_combined])

            # Add FFT if used
            if self.use_fft:
                # Clean signals
                signals_clean = np.nan_to_num(signals_np, nan=0.0, posinf=0.0, neginf=0.0)

                # Normalize
                signals_mean = signals_clean.mean(axis=1, keepdims=True)
                signals_std = signals_clean.std(axis=1, keepdims=True)
                signals_std[signals_std < 1e-10] = 1.0
                signals_normalized = (signals_clean - signals_mean) / signals_std

                # Compute FFT
                fft_features = np.abs(np.fft.rfft(signals_normalized, axis=1))

                # Clean FFT result
                fft_features = np.nan_to_num(fft_features, nan=0.0, posinf=0.0, neginf=0.0)
                fft_features = np.clip(fft_features, -1e10, 1e10)
                fft_features = fft_features.squeeze()
                features = np.hstack([features, fft_features])

            # Predict
            preds = self.classifier.predict(features)
            all_preds.append(preds)

        predictions = np.concatenate(all_preds)

        if all_labels:
            true_labels = np.concatenate(all_labels)
            return predictions, true_labels
        else:
            return predictions
if __name__ == "__main__":
    from utils import create_dataloader_from_arff
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    train_dataloader, train_mean, train_std = create_dataloader_from_arff(
        arff_file_path='/Users/hxh/PycharmProjects/wavelet/wavelet/Dataset/WormsTwoClass/WormsTwoClass_TRAIN.arff',
        batch_size=32,
        shuffle=True
    )
    # 测试集：使用训练集的参数进行标准化
    test_dataloader, _, _ = create_dataloader_from_arff(
        arff_file_path='/Users/hxh/PycharmProjects/wavelet/wavelet/Dataset/WormsTwoClass/WormsTwoClass_TEST.arff',
        batch_size=32,
        shuffle=False,
        mean=train_mean,
        std=train_std
    )

    model = MultiModelParallelClassifier(
        filter_lengths=[3, 7, 11, 15],
        levels=3,
        signal_length=900,
        num_classes=2,

        # ===== 所有特征开关 =====
        use_wavelet=True,  # ← 小波 on/off
        use_pca=True,  # ← PCA on/off
        use_stats=False,  # ← 统计特征 on/off
        use_fft=True,  # ← FFT on/off

        # ===== 训练参数 =====
        max_epochs=2,
        initial_lr=0.001,
        patience=10
    )
    model.train_models_parallel(train_dataloader, device='cpu')

    model.train_classifier(train_dataloader)

    # print('training data')
    # preds, labels = model.predict(train_dataloader)
    # cm = confusion_matrix(labels, preds)
    # # 方法 1: 打印数值
    # print("Confusion Matrix:")
    # print(cm)
    print('test')
    preds, labels = model.predict(test_dataloader)
    cm = confusion_matrix(labels, preds)

    # 方法 1: 打印数值
    print("Confusion Matrix:")
    print(cm)