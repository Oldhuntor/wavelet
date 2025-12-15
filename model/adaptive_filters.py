import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnableWaveletTransform(nn.Module):
    def __init__(self, filter_length, levels, init_type='random', use_frequency_constraint=True,
                 use_learnable_activation=False):
        super().__init__()
        self.filter_length = filter_length
        self.levels = levels
        self.use_frequency_constraint = use_frequency_constraint
        self.use_learnable_activation = use_learnable_activation

        # Initialize filters based on type
        if init_type == 'haar':
            # Haar wavelet
            low_pass_init = torch.tensor([1.0, 1.0]) / np.sqrt(2)
            high_pass_init = torch.tensor([1.0, -1.0]) / np.sqrt(2)
            if filter_length != 2:
                print(f"Warning: Haar requires filter_length=2, but got {filter_length}. Using 2.")
                self.filter_length = 2
        elif init_type == 'db4':
            # Daubechies-4 wavelet
            low_pass_init = torch.tensor([
                0.6830127, 1.1830127, 0.3169873, -0.1830127
            ]) / np.sqrt(2)
            high_pass_init = torch.tensor([
                -0.1830127, -0.3169873, 1.1830127, -0.6830127
            ]) / np.sqrt(2)
            if filter_length != 4:
                print(f"Warning: DB4 requires filter_length=4, but got {filter_length}. Using 4.")
                self.filter_length = 4
        else:
            # Random initialization
            low_pass_init = torch.randn(filter_length)
            high_pass_init = torch.randn(filter_length)

        self.low_pass = nn.Parameter(low_pass_init)
        self.high_pass = nn.Parameter(high_pass_init)

        # Normalize initial filters
        with torch.no_grad():
            self.low_pass.data = F.normalize(self.low_pass.data, dim=0)
            self.high_pass.data = F.normalize(self.high_pass.data, dim=0)

        # Learnable activation between levels (soft thresholding)
        if use_learnable_activation:
            # threshold and sharpness parameters for each level
            self.threshold_params = nn.ParameterList([
                nn.Parameter(torch.tensor([0.5, 10.0]))  # [threshold, sharpness]
                for _ in range(levels)
            ])

    def apply_frequency_constraint(self, filter_param, filter_type='low'):
        """Apply frequency domain constraint to filter

        Args:
            filter_param: Filter tensor
            filter_type: 'low' for low-pass, 'high' for high-pass

        Returns:
            Constrained filter
        """
        # Compute FFT
        fft_size = max(64, self.filter_length * 4)  # Zero-pad for better frequency resolution
        filter_padded = F.pad(filter_param, (0, fft_size - self.filter_length))
        filter_fft = torch.fft.rfft(filter_padded)
        magnitude = torch.abs(filter_fft)
        phase = torch.angle(filter_fft)

        # Create frequency weights
        freqs = torch.linspace(0, 1, len(magnitude), device=filter_param.device)

        if filter_type == 'low':
            # Low-pass: emphasize low frequencies, suppress high frequencies
            weights = torch.exp(-3.0 * freqs)  # Exponential decay
        else:
            # High-pass: suppress low frequencies, emphasize high frequencies
            weights = 1.0 - torch.exp(-3.0 * freqs)  # Inverse of low-pass

        # Apply constraint to magnitude
        constrained_magnitude = magnitude * weights

        # Reconstruct filter
        constrained_fft = constrained_magnitude * torch.exp(1j * phase)
        constrained_filter = torch.fft.irfft(constrained_fft, n=fft_size)

        # Trim back to original length
        constrained_filter = constrained_filter[:self.filter_length]

        # Normalize
        constrained_filter = F.normalize(constrained_filter, dim=0)

        return constrained_filter

    def get_constrained_filters(self):
        """Get filters with frequency constraints applied"""
        if self.use_frequency_constraint:
            low_pass = self.apply_frequency_constraint(self.low_pass, filter_type='low')
            high_pass = self.apply_frequency_constraint(self.high_pass, filter_type='high')
        else:
            low_pass = F.normalize(self.low_pass, dim=0)
            high_pass = F.normalize(self.high_pass, dim=0)

        return low_pass, high_pass

    def dwt_single(self, signal, low_pass, high_pass):
        """Single level DWT"""
        batch_size = signal.shape[0]

        # Reshape for conv1d: (batch, 1, length)
        signal = signal.unsqueeze(1)
        low_pass = low_pass.view(1, 1, -1)
        high_pass = high_pass.view(1, 1, -1)

        # Convolve
        approx = F.conv1d(signal, low_pass, padding=self.filter_length - 1)
        detail = F.conv1d(signal, high_pass, padding=self.filter_length - 1)

        # Downsample
        approx = approx[:, :, ::2]
        detail = detail[:, :, ::2]

        return approx.squeeze(1), detail.squeeze(1)

    def dwt_multilevel(self, signal):
        """Multi-level DWT"""
        coeffs = []
        current = signal

        # Get constrained filters
        low_pass, high_pass = self.get_constrained_filters()

        for level in range(self.levels):
            approx, detail = self.dwt_single(current, low_pass, high_pass)
            coeffs.append(detail)

            # Apply soft learnable threshold between levels
            if self.use_learnable_activation and level < self.levels - 1:
                # Soft thresholding: shrinks values below threshold smoothly
                # f(x) = x * sigmoid(k * (|x| - t))
                # where t = threshold, k = sharpness
                threshold, sharpness = self.threshold_params[level]
                magnitude = torch.abs(approx)
                # Smooth step function via sigmoid
                gate = torch.sigmoid(sharpness * (magnitude - threshold))
                approx = approx * gate

            current = approx

        coeffs.append(current)
        return coeffs[::-1]  # [approx_N, detail_N, ..., detail_1]

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
        low_pass = low_pass.flip(0).view(1, 1, -1)
        high_pass = high_pass.flip(0).view(1, 1, -1)

        # Convolve
        rec_approx = F.conv1d(up_approx, low_pass, padding=self.filter_length - 1)
        rec_detail = F.conv1d(up_detail, high_pass, padding=self.filter_length - 1)

        # Trim
        start = self.filter_length - 1
        min_len = min(rec_approx.shape[2], rec_detail.shape[2])
        reconstructed = (rec_approx[:, :, :min_len] + rec_detail[:, :, :min_len])[:, :, start:start + target_len]

        return reconstructed.squeeze(1)

    def idwt_multilevel(self, coeffs, signal_len):
        """Multi-level IDWT"""
        levels = len(coeffs) - 1
        current = coeffs[0]

        # Get constrained filters
        low_pass, high_pass = self.get_constrained_filters()

        for i in range(levels):
            detail = coeffs[i + 1]
            target_len = min(current.shape[1] * 2, signal_len)
            current = self.idwt_single(current, detail, low_pass, high_pass, target_len)

        return current[:, :signal_len]

    def forward(self, signal):
        """Forward pass: decompose signal"""
        return self.dwt_multilevel(signal)


class WaveletClassifier(nn.Module):
    def __init__(self, filter_length, levels, signal_length, num_classes, hidden_dim=128,
                 init_type='random', use_frequency_constraint=True, use_learnable_activation=False):
        super().__init__()
        self.wavelet = LearnableWaveletTransform(filter_length, levels, init_type,
                                                 use_frequency_constraint, use_learnable_activation)
        self.signal_length = signal_length
        self.levels = levels
        self.filter_length = self.wavelet.filter_length  # Use actual filter length after init

        # Calculate total coefficient dimension
        total_dim = self._calculate_coeff_dim()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def _calculate_coeff_dim(self):
        """Calculate total flattened coefficient dimension"""
        total_dim = 0
        temp_len = self.signal_length

        for _ in range(self.levels):
            # After convolution with 'full' mode and downsampling
            conv_len = temp_len + self.filter_length - 1
            temp_len = (conv_len + 1) // 2  # Downsample by 2
            total_dim += temp_len

        total_dim += temp_len  # Final approximation
        return total_dim

    def forward(self, x):
        # Get wavelet coefficients
        coeffs = self.wavelet(x)

        # Flatten all coefficients
        flat_coeffs = torch.cat([c.flatten(1) for c in coeffs], dim=1)

        # Classify
        logits = self.classifier(flat_coeffs)

        return logits, coeffs

    def compute_reconstruction_loss(self, signal, coeffs):
        """Compute reconstruction error as regularization"""
        reconstructed = self.wavelet.idwt_multilevel(coeffs, self.signal_length)
        recon_loss = F.mse_loss(reconstructed, signal)
        return recon_loss


# Training example
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    signal_length = 128
    num_classes = 5
    levels = 3

    # Dummy data
    signals = torch.randn(batch_size, signal_length)
    labels = torch.randint(0, num_classes, (batch_size,))

    print("=" * 60)
    print("Test 1: WITHOUT Soft Threshold")
    print("=" * 60)
    model_no_act = WaveletClassifier(8, levels, signal_length, num_classes,
                                     init_type='db4', use_frequency_constraint=True,
                                     use_learnable_activation=False)

    logits, coeffs = model_no_act(signals)
    classification_loss = F.cross_entropy(logits, labels)
    reconstruction_loss = model_no_act.compute_reconstruction_loss(signals, coeffs)
    total_loss = classification_loss + 0.1 * reconstruction_loss

    print(f"Classification loss: {classification_loss.item():.4f}")
    print(f"Reconstruction loss: {reconstruction_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")

    print("\n" + "=" * 60)
    print("Test 2: WITH Soft Learnable Threshold")
    print("=" * 60)
    model_with_act = WaveletClassifier(8, levels, signal_length, num_classes,
                                       init_type='db4', use_frequency_constraint=True,
                                       use_learnable_activation=True)

    print("Initial threshold params (threshold, sharpness) per level:")
    for i, params in enumerate(model_with_act.wavelet.threshold_params):
        print(f"  Level {i + 1}: threshold={params[0].item():.3f}, sharpness={params[1].item():.3f}")

    logits, coeffs = model_with_act(signals)
    classification_loss = F.cross_entropy(logits, labels)
    reconstruction_loss = model_with_act.compute_reconstruction_loss(signals, coeffs)
    total_loss = classification_loss + 0.1 * reconstruction_loss

    print(f"\nClassification loss: {classification_loss.item():.4f}")
    print(f"Reconstruction loss: {reconstruction_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")

    print("\n" + "=" * 60)
    print("Test 3: Training with Soft Threshold (10 steps)")
    print("=" * 60)

    model = WaveletClassifier(8, levels, signal_length, num_classes,
                              init_type='random', use_frequency_constraint=True,
                              use_learnable_activation=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training...")
    for step in range(10):
        optimizer.zero_grad()
        logits, coeffs = model(signals)
        classification_loss = F.cross_entropy(logits, labels)
        reconstruction_loss = model.compute_reconstruction_loss(signals, coeffs)
        total_loss = classification_loss + 0.1 * reconstruction_loss
        total_loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print(f"\nStep {step}: Loss={total_loss.item():.4f}")
            for i, params in enumerate(model.wavelet.threshold_params):
                print(f"  Level {i + 1}: threshold={params[0].item():.3f}, sharpness={params[1].item():.3f}")

    print("\n" + "=" * 60)
    print("Soft threshold explanation:")
    print("- threshold: values below this are smoothly suppressed")
    print("- sharpness: controls how abrupt the transition is")
    print("- Higher sharpness → more like hard threshold")
    print("- Lower sharpness → more gradual suppression")
    print("=" * 60)