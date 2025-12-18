"""
Multi-Wavelet Transform: Multiple learnable DWT filter pairs
=============================================================

Architecture: N pairs of (low_pass, high_pass) filters
Each pair performs standard DWT independently
Combine all features for classification

Training:
  Stage 1: Train all filter pairs (reconstruction loss)
  Stage 2: Train classifier on combined features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class MultiWaveletTransform(nn.Module):
    """Multiple learnable DWT filter pairs

    Instead of N independent filters, use N pairs of (low_pass, high_pass)
    Each pair performs standard DWT, then combine all features
    """

    def __init__(self, filter_length, levels, num_wavelets=4, init_type='random',
                 use_frequency_constraint=True, use_learnable_activation=False):
        super().__init__()
        self.filter_length = filter_length
        self.levels = levels
        self.num_wavelets = num_wavelets
        self.use_frequency_constraint = use_frequency_constraint
        self.use_learnable_activation = use_learnable_activation

        # Multiple filter pairs
        self.low_pass_filters = nn.ParameterList()
        self.high_pass_filters = nn.ParameterList()

        for i in range(num_wavelets):
            if init_type == 'random':
                low_pass = torch.randn(filter_length)
                high_pass = torch.randn(filter_length)
            elif init_type == 'frequency_init':
                # Initialize each pair for different frequency bands
                center_freq = (i + 0.5) / num_wavelets
                low_pass = torch.randn(filter_length)
                high_pass = torch.randn(filter_length)

                low_pass = self._init_frequency_constraint(low_pass, center_freq, 'low')
                high_pass = self._init_frequency_constraint(high_pass, center_freq, 'high')
            else:
                low_pass = torch.randn(filter_length)
                high_pass = torch.randn(filter_length)

            with torch.no_grad():
                low_pass = F.normalize(low_pass, dim=0) * np.sqrt(2)
                high_pass = F.normalize(high_pass, dim=0) * np.sqrt(2)

            self.low_pass_filters.append(nn.Parameter(low_pass))
            self.high_pass_filters.append(nn.Parameter(high_pass))

        if use_learnable_activation:
            self.threshold_params = nn.ModuleList([
                nn.ParameterList([
                    nn.Parameter(torch.tensor([0.5, 10.0])) for _ in range(levels)
                ]) for _ in range(num_wavelets)
            ])

    def _init_frequency_constraint(self, filter_param, center_freq, filter_type):
        """Apply frequency constraint during initialization"""
        fft_size = max(64, self.filter_length * 4)
        filter_padded = F.pad(filter_param, (0, fft_size - self.filter_length))
        filter_fft = torch.fft.rfft(filter_padded)
        magnitude = torch.abs(filter_fft)
        phase = torch.angle(filter_fft)

        freqs = torch.linspace(0, 1, len(magnitude))

        if filter_type == 'low':
            weights = torch.exp(-((freqs - center_freq * 0.5) / 0.3) ** 2)
        else:
            weights = torch.exp(-((freqs - (center_freq * 0.5 + 0.5)) / 0.3) ** 2)

        constrained_magnitude = magnitude * weights
        constrained_fft = constrained_magnitude * torch.exp(1j * phase)
        constrained_filter = torch.fft.irfft(constrained_fft, n=fft_size)

        return constrained_filter[:self.filter_length]

    def apply_frequency_constraint(self, filter_param, wavelet_idx, filter_type='low'):
        """Apply frequency constraint per wavelet pair"""
        if not self.use_frequency_constraint:
            return F.normalize(filter_param, dim=0) * np.sqrt(2)

        fft_size = max(64, self.filter_length * 4)
        filter_padded = F.pad(filter_param, (0, fft_size - self.filter_length))
        filter_fft = torch.fft.rfft(filter_padded)
        magnitude = torch.abs(filter_fft)
        phase = torch.angle(filter_fft)

        freqs = torch.linspace(0, 1, len(magnitude), device=filter_param.device)

        # Each pair focuses on different frequency band
        center = (wavelet_idx + 0.5) / self.num_wavelets
        width = 1.0 / self.num_wavelets

        if filter_type == 'low':
            target_center = center * 0.5
        else:
            target_center = center * 0.5 + 0.5

        weights = torch.exp(-((freqs - target_center) / width) ** 2)

        constrained_magnitude = magnitude * weights
        constrained_fft = constrained_magnitude * torch.exp(1j * phase)
        constrained_filter = torch.fft.irfft(constrained_fft, n=fft_size)
        constrained_filter = constrained_filter[:self.filter_length]

        return F.normalize(constrained_filter, dim=0) * np.sqrt(2)

    def get_constrained_filters(self, wavelet_idx):
        """Get constrained low/high pass for specific pair"""
        low_pass = self.apply_frequency_constraint(
            self.low_pass_filters[wavelet_idx], wavelet_idx, filter_type='low'
        )
        high_pass = self.apply_frequency_constraint(
            self.high_pass_filters[wavelet_idx], wavelet_idx, filter_type='high'
        )
        return low_pass, high_pass

    def dwt_single(self, signal, low_pass, high_pass):
        """Single level DWT"""
        signal = signal.unsqueeze(1)
        low_pass = low_pass.view(1, 1, -1)
        high_pass = high_pass.view(1, 1, -1)

        approx = F.conv1d(signal, low_pass, padding=self.filter_length - 1)
        detail = F.conv1d(signal, high_pass, padding=self.filter_length - 1)

        approx = approx[:, :, ::2]
        detail = detail[:, :, ::2]

        return approx.squeeze(1), detail.squeeze(1)

    def dwt_multilevel_single_pair(self, signal, wavelet_idx):
        """Multi-level DWT for one pair"""
        coeffs = []
        current = signal

        low_pass, high_pass = self.get_constrained_filters(wavelet_idx)

        for level in range(self.levels):
            approx, detail = self.dwt_single(current, low_pass, high_pass)
            coeffs.append(detail)

            if self.use_learnable_activation and level < self.levels - 1:
                threshold, sharpness = self.threshold_params[wavelet_idx][level]
                magnitude = torch.abs(approx)
                gate = torch.sigmoid(sharpness * (magnitude - threshold))
                approx = approx * gate

            current = approx

        coeffs.append(current)
        return coeffs[::-1]

    def forward(self, signal):
        """Apply all wavelet pairs

        Returns:
            List of coefficient lists, one per wavelet pair
            all_coeffs[i] = coeffs from pair i
        """

        all_coeffs = []

        for i in range(self.num_wavelets):
            coeffs = self.dwt_multilevel_single_pair(signal, i)
            all_coeffs.append(coeffs)

        return all_coeffs

    def idwt_single(self, approx, detail, low_pass, high_pass, target_len):
        """Single level IDWT"""
        batch_size = approx.shape[0]
        max_len = max(approx.shape[1], detail.shape[1])

        up_approx = torch.zeros(batch_size, max_len * 2, device=approx.device)
        up_approx[:, ::2] = approx if approx.shape[1] == max_len else F.pad(approx, (0, max_len - approx.shape[1]))

        up_detail = torch.zeros(batch_size, max_len * 2, device=detail.device)
        up_detail[:, ::2] = detail if detail.shape[1] == max_len else F.pad(detail, (0, max_len - detail.shape[1]))

        up_approx = up_approx.unsqueeze(1)
        up_detail = up_detail.unsqueeze(1)
        low_pass = low_pass.flip(0).view(1, 1, -1)
        high_pass = high_pass.flip(0).view(1, 1, -1)

        rec_approx = F.conv1d(up_approx, low_pass, padding=self.filter_length - 1)
        rec_detail = F.conv1d(up_detail, high_pass, padding=self.filter_length - 1)

        start = self.filter_length - 1
        min_len = min(rec_approx.shape[2], rec_detail.shape[2])
        reconstructed = (rec_approx[:, :, :min_len] + rec_detail[:, :, :min_len])[:, :, start:start + target_len]

        return reconstructed.squeeze(1)

    def idwt_multilevel_single_pair(self, coeffs, signal_len, wavelet_idx):
        """Multi-level IDWT for one pair"""
        levels = len(coeffs) - 1
        current = coeffs[0]

        low_pass, high_pass = self.get_constrained_filters(wavelet_idx)

        for i in range(levels):
            detail = coeffs[i + 1]
            target_len = min(current.shape[1] * 2, signal_len)
            current = self.idwt_single(current, detail, low_pass, high_pass, target_len)

        return current[:, :signal_len]

    def compute_reconstruction_loss(self, signal, all_coeffs):
        """Reconstruction loss for all pairs (average)"""
        total_loss = 0.0

        for i in range(self.num_wavelets):
            reconstructed = self.idwt_multilevel_single_pair(all_coeffs[i], signal.shape[1], i)
            loss = F.mse_loss(reconstructed, signal)
            total_loss += loss

        return total_loss / self.num_wavelets


class MultiWaveletClassifier:
    """Multi-wavelet DWT + Classifier

    Training:
      Stage 1: Train all filter pairs (reconstruction)
      Stage 2: Combine features, train classifier
    """

    def __init__(self, filter_length, levels, signal_length, num_classes, num_wavelets=4,
                 init_type='random', use_frequency_constraint=True, use_learnable_activation=False,
                 n_estimators=200, max_depth=None, random_state=42):

        self.wavelet = MultiWaveletTransform(
            filter_length, levels, num_wavelets, init_type,
            use_frequency_constraint, use_learnable_activation
        )
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.num_wavelets = num_wavelets

        self.classifier = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.classifier_fitted = False

    def extract_features(self, signals, device='cpu'):
        """Extract features from all wavelet pairs"""
        self.wavelet = self.wavelet.to(device)
        self.wavelet.eval()

        with torch.no_grad():
            all_coeffs = self.wavelet(signals)

            # Flatten all coefficients from all pairs
            all_features = []
            for pair_idx in range(self.num_wavelets):
                coeffs = all_coeffs[pair_idx]
                flat = torch.cat([c.flatten(1) for c in coeffs], dim=1)
                all_features.append(flat)

            # Concatenate features from all pairs
            combined_features = torch.cat(all_features, dim=1)

        return combined_features.cpu().numpy()

    def train_filters(self, train_loader, epochs=30, lr=0.001, device='cpu'):
        """Stage 1: Train all filter pairs"""
        print("\nStage 1: Training all filter pairs")
        print("=" * 70)

        self.wavelet = self.wavelet.to(device)
        self.wavelet.train()

        optimizer = torch.optim.Adam(self.wavelet.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for signals, _ in train_loader:
                signals = signals.to(device)
                optimizer.zero_grad()
                signals.squeeze_()
                all_coeffs = self.wavelet(signals)
                loss = self.wavelet.compute_reconstruction_loss(signals, all_coeffs)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}: Loss={epoch_loss / len(train_loader):.6f}")

        # Freeze all filters
        for param in self.wavelet.parameters():
            param.requires_grad = False
        self.wavelet.eval()

        print(f"✓ All {self.num_wavelets} filter pairs trained")

    def train_classifier(self, train_loader, device='cpu'):
        """Stage 2: Train classifier on combined features"""
        print("\nStage 2: Training classifier on combined features")
        print("=" * 70)

        all_features = []
        all_labels = []

        for signals, labels in train_loader:
            signals = signals.to(device)
            signals.squeeze_()
            features = self.extract_features(signals, device)
            all_features.append(features)
            all_labels.append(labels.numpy())

        X_train = np.vstack(all_features)
        y_train = np.concatenate(all_labels)

        print(f"Combined features: {X_train.shape}")

        self.classifier.fit(X_train, y_train)
        self.classifier_fitted = True

        print(f"✓ Classifier trained")

    def predict(self, test_loader, device='cpu'):
        """Predict on test_loader

        Returns:
            predictions: numpy array
            labels: numpy array (if test_loader has labels)
        """
        all_preds = []
        all_labels = []

        for batch in test_loader:
            if len(batch) == 2:
                signals, labels = batch
                all_labels.append(labels.numpy())
            else:
                signals = batch

            signals = signals.to(device)
            signals.squeeze_()
            features = self.extract_features(signals, device)
            preds = self.classifier.predict(features)
            all_preds.append(preds)

        predictions = np.concatenate(all_preds)

        if all_labels:
            return predictions, np.concatenate(all_labels)
        else:
            return predictions


if __name__ == "__main__":

    from utils import create_dataloader_from_arff

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

    model = MultiWaveletClassifier(
        filter_length=8,
        levels=4,
        signal_length=900,
        num_classes=2,
        num_wavelets=4,
        use_frequency_constraint=False,
        use_learnable_activation=False,
        init_type='frequency_init'
    )

    model.train_filters(train_dataloader, epochs=100, lr=0.01, device='cpu')
    model.train_classifier(train_dataloader)
    print('training data')
    preds, labels = model.predict(train_dataloader)
    cm = confusion_matrix(labels, preds)
    # 方法 1: 打印数值
    print("Confusion Matrix:")
    print(cm)
    print('test')
    preds, labels = model.predict(test_dataloader)
    cm = confusion_matrix(labels, preds)

    # 方法 1: 打印数值
    print("Confusion Matrix:")
    print(cm)