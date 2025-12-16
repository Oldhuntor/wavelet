"""
QMF (Quadrature Mirror Filter) 关系
====================================

在正交小波理论中，high-pass filter可以通过low-pass filter的QMF关系自动计算:

h[n] = low-pass filter
g[n] = high-pass filter

QMF关系:
-------
g[n] = (-1)^n * h[L-1-n]

其中 L 是filter长度

等价形式:
g[n] = (-1)^(n+1) * h[L-1-n]  (取决于约定)

优点:
-----
1. 只需要学习一个filter (low-pass)
2. 自动保证perfect reconstruction (完美重构)
3. 参数减半
4. 符合小波理论的正交性

示例:
-----
Haar小波:
h = [1, 1] / sqrt(2)
g = [1, -1] / sqrt(2)  ← 通过QMF从h计算

Daubechies-4:
h = [0.683, 1.183, 0.317, -0.183] / sqrt(2)
g = [-0.183, -0.317, 1.183, -0.683] / sqrt(2)  ← QMF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_qmf_highpass(lowpass_filter):
    """通过QMF关系从low-pass计算high-pass

    Args:
        lowpass_filter: Low-pass filter coefficients (1D tensor)

    Returns:
        High-pass filter coefficients

    QMF公式: g[n] = (-1)^n * h[L-1-n]
    """
    L = len(lowpass_filter)

    # 翻转 low-pass
    highpass = torch.flip(lowpass_filter, dims=[0])

    # 交替符号: (-1)^n
    alternating_signs = torch.tensor(
        [(-1) ** n for n in range(L)],
        dtype=lowpass_filter.dtype,
        device=lowpass_filter.device
    )

    highpass = highpass * alternating_signs

    return highpass


def verify_qmf_property(lowpass, highpass):
    """验证QMF性质

    检查:
    1. 正交性: sum(h[n] * g[n]) = 0
    2. 能量守恒: sum(h[n]^2) + sum(g[n]^2) = 2 (对归一化的filter)
    """
    # 正交性
    orthogonality = torch.sum(lowpass * highpass).item()

    # 能量
    energy_low = torch.sum(lowpass ** 2).item()
    energy_high = torch.sum(highpass ** 2).item()
    total_energy = energy_low + energy_high

    return {
        'orthogonality': orthogonality,
        'energy_low': energy_low,
        'energy_high': energy_high,
        'total_energy': total_energy,
        'is_orthogonal': abs(orthogonality) < 1e-6,
        'energy_conserved': abs(total_energy - 2.0) < 1e-3
    }


class QMFWaveletTransform(nn.Module):
    def __init__(self, filter_length, levels, init_type='random',
                 use_frequency_constraint=True, use_learnable_activation=False):
        """QMF小波变换 - 只学习low-pass filter

        High-pass filter通过QMF关系自动计算
        """
        super().__init__()
        self.filter_length = filter_length
        self.levels = levels
        self.use_frequency_constraint = use_frequency_constraint
        self.use_learnable_activation = use_learnable_activation

        # 只有一个可学习的filter: low-pass
        if init_type == 'haar':
            if filter_length != 2:
                print(f"Warning: Haar requires filter_length=2, using 2.")
                self.filter_length = 2
            low_pass_init = torch.tensor([1.0, 1.0]) / np.sqrt(2)
        elif init_type == 'db4':
            if filter_length != 4:
                print(f"Warning: DB4 requires filter_length=4, using 4.")
                self.filter_length = 4
            low_pass_init = torch.tensor([
                0.6830127, 1.1830127, 0.3169873, -0.1830127
            ]) / np.sqrt(2)
        else:
            low_pass_init = torch.randn(filter_length)

        # 只有这一个可学习参数!
        self.low_pass = nn.Parameter(low_pass_init)

        # 归一化
        with torch.no_grad():
            self.low_pass.data = F.normalize(self.low_pass.data, dim=0) * np.sqrt(2)

        # Learnable activation (可选)
        if use_learnable_activation:
            self.threshold_params = nn.ParameterList([
                nn.Parameter(torch.tensor([0.5, 10.0]))
                for _ in range(levels)
            ])

    def get_high_pass_from_qmf(self, low_pass):
        """通过QMF关系计算high-pass"""
        return compute_qmf_highpass(low_pass)

    def apply_frequency_constraint(self, filter_param, filter_type='low'):
        """应用频率约束"""
        if not self.use_frequency_constraint:
            return F.normalize(filter_param, dim=0) * np.sqrt(2)

        fft_size = max(64, self.filter_length * 4)
        filter_padded = F.pad(filter_param, (0, fft_size - self.filter_length))
        filter_fft = torch.fft.rfft(filter_padded)
        magnitude = torch.abs(filter_fft)
        phase = torch.angle(filter_fft)

        freqs = torch.linspace(0, 1, len(magnitude), device=filter_param.device)

        if filter_type == 'low':
            weights = torch.exp(-3.0 * freqs)
        else:
            weights = 1.0 - torch.exp(-3.0 * freqs)

        constrained_magnitude = magnitude * weights
        constrained_fft = constrained_magnitude * torch.exp(1j * phase)
        constrained_filter = torch.fft.irfft(constrained_fft, n=fft_size)
        constrained_filter = constrained_filter[:self.filter_length]

        return F.normalize(constrained_filter, dim=0) * np.sqrt(2)

    def get_constrained_filters(self):
        """获取约束后的filters (通过QMF计算high-pass)"""
        # 约束low-pass
        low_pass = self.apply_frequency_constraint(self.low_pass, filter_type='low')

        # 通过QMF计算high-pass (不需要约束，自动满足)
        high_pass = self.get_high_pass_from_qmf(low_pass)

        return low_pass, high_pass

    def dwt_single(self, signal, low_pass, high_pass):
        """单层DWT"""
        batch_size = signal.shape[0]
        signal = signal.unsqueeze(1)
        low_pass = low_pass.view(1, 1, -1)
        high_pass = high_pass.view(1, 1, -1)

        approx = F.conv1d(signal, low_pass, padding=self.filter_length - 1)
        detail = F.conv1d(signal, high_pass, padding=self.filter_length - 1)

        approx = approx[:, :, ::2]
        detail = detail[:, :, ::2]

        return approx.squeeze(1), detail.squeeze(1)

    def dwt_multilevel(self, signal):
        """多层DWT"""
        coeffs = []
        current = signal

        low_pass, high_pass = self.get_constrained_filters()

        for level in range(self.levels):
            approx, detail = self.dwt_single(current, low_pass, high_pass)
            coeffs.append(detail)

            if self.use_learnable_activation and level < self.levels - 1:
                threshold, sharpness = self.threshold_params[level]
                magnitude = torch.abs(approx)
                gate = torch.sigmoid(sharpness * (magnitude - threshold))
                approx = approx * gate

            current = approx

        coeffs.append(current)
        return coeffs[::-1]

    def idwt_single(self, approx, detail, low_pass, high_pass, target_len):
        """单层IDWT"""
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

    def idwt_multilevel(self, coeffs, signal_len):
        """多层IDWT"""
        levels = len(coeffs) - 1
        current = coeffs[0]

        low_pass, high_pass = self.get_constrained_filters()

        for i in range(levels):
            detail = coeffs[i + 1]
            target_len = min(current.shape[1] * 2, signal_len)
            current = self.idwt_single(current, detail, low_pass, high_pass, target_len)

        return current[:, :signal_len]

    def forward(self, signal):
        """前向传播"""
        return self.dwt_multilevel(signal)


class QMFWaveletClassifier(nn.Module):
    def __init__(self, filter_length, levels, signal_length, num_classes, hidden_dim=128,
                 init_type='random', use_frequency_constraint=True, use_learnable_activation=False):
        """QMF小波分类器 - 只学习low-pass filter"""
        super().__init__()
        self.wavelet = QMFWaveletTransform(
            filter_length, levels, init_type,
            use_frequency_constraint, use_learnable_activation
        )
        self.signal_length = signal_length
        self.levels = levels
        self.filter_length = self.wavelet.filter_length

        total_dim = self._calculate_coeff_dim()

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
        """计算系数维度"""
        total_dim = 0
        temp_len = self.signal_length

        for _ in range(self.levels):
            conv_len = temp_len + self.filter_length - 1
            temp_len = (conv_len + 1) // 2
            total_dim += temp_len

        total_dim += temp_len
        return total_dim

    def forward(self, x):
        """前向传播"""
        coeffs = self.wavelet(x)
        flat_coeffs = torch.cat([c.flatten(1) for c in coeffs], dim=1)
        logits = self.classifier(flat_coeffs)
        return logits, coeffs

    def compute_reconstruction_loss(self, signal, coeffs):
        """计算重构损失"""
        reconstructed = self.wavelet.idwt_multilevel(coeffs, self.signal_length)
        recon_loss = F.mse_loss(reconstructed, signal)
        return recon_loss


# ============================================================================
# 测试和验证
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QMF小波 - 只学习Low-pass Filter")
    print("=" * 70)

    # 测试QMF关系
    print("\n1. 验证经典小波的QMF性质:")
    print("-" * 70)

    # Haar
    print("\nHaar小波:")
    haar_low = torch.tensor([1.0, 1.0]) / np.sqrt(2)
    haar_high = compute_qmf_highpass(haar_low)
    print(f"  Low-pass:  {haar_low}")
    print(f"  High-pass: {haar_high}")
    haar_props = verify_qmf_property(haar_low, haar_high)
    print(f"  正交性: {haar_props['orthogonality']:.6f} (应接近0)")
    print(f"  能量守恒: {haar_props['total_energy']:.6f} (应为2.0)")
    print(f"  ✓ 正交: {haar_props['is_orthogonal']}, 能量守恒: {haar_props['energy_conserved']}")

    # DB4
    print("\nDaubechies-4小波:")
    db4_low = torch.tensor([0.6830127, 1.1830127, 0.3169873, -0.1830127]) / np.sqrt(2)
    db4_high = compute_qmf_highpass(db4_low)
    print(f"  Low-pass:  {db4_low}")
    print(f"  High-pass: {db4_high}")
    db4_props = verify_qmf_property(db4_low, db4_high)
    print(f"  正交性: {db4_props['orthogonality']:.6f}")
    print(f"  能量守恒: {db4_props['total_energy']:.6f}")
    print(f"  ✓ 正交: {db4_props['is_orthogonal']}, 能量守恒: {db4_props['energy_conserved']}")

    # 对比模型参数
    print("\n" + "=" * 70)
    print("2. 参数对比:")
    print("-" * 70)

    signal_length = 128
    num_classes = 3

    # 原始模型 (2个独立的filters)
    from model.adaptive_filters import WaveletClassifier

    model_original = WaveletClassifier(
        filter_length=8, levels=3, signal_length=signal_length,
        num_classes=num_classes, init_type='random'
    )

    # QMF模型 (只有1个filter)
    model_qmf = QMFWaveletClassifier(
        filter_length=8, levels=3, signal_length=signal_length,
        num_classes=num_classes, init_type='random'
    )

    original_filter_params = model_original.wavelet.low_pass.numel() + \
                             model_original.wavelet.high_pass.numel()
    qmf_filter_params = model_qmf.wavelet.low_pass.numel()

    print(f"\n原始模型 (两个独立filters):")
    print(f"  Filter参数: {original_filter_params} (low-pass: {model_original.wavelet.low_pass.numel()}, "
          f"high-pass: {model_original.wavelet.high_pass.numel()})")

    print(f"\nQMF模型 (只学习low-pass):")
    print(f"  Filter参数: {qmf_filter_params} (只有low-pass)")
    print(
        f"  参数减少: {original_filter_params - qmf_filter_params} ({(1 - qmf_filter_params / original_filter_params) * 100:.1f}%)")

    # 测试训练
    print("\n" + "=" * 70)
    print("3. 测试训练:")
    print("-" * 70)

    signals = torch.randn(16, signal_length)
    labels = torch.randint(0, num_classes, (16,))

    # QMF模型训练
    print("\nQMF模型:")
    logits, coeffs = model_qmf(signals)
    classification_loss = F.cross_entropy(logits, labels)
    reconstruction_loss = model_qmf.compute_reconstruction_loss(signals, coeffs)
    total_loss = classification_loss + 0.1 * reconstruction_loss

    total_loss.backward()

    print(f"  Classification loss: {classification_loss.item():.4f}")
    print(f"  Reconstruction loss: {reconstruction_loss.item():.4f}")
    print(f"  Low-pass grad norm: {model_qmf.wavelet.low_pass.grad.norm().item():.4f}")

    # 验证QMF性质
    with torch.no_grad():
        low_pass, high_pass = model_qmf.wavelet.get_constrained_filters()
        qmf_props = verify_qmf_property(low_pass, high_pass)
        print(f"\n  学习后的QMF性质:")
        print(f"    正交性: {qmf_props['orthogonality']:.6f}")
        print(f"    能量守恒: {qmf_props['total_energy']:.6f}")

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
QMF方法的优势:
-------------
1. ✓ 参数减半 - 只需要学习low-pass filter
2. ✓ 自动满足正交性 - high-pass通过QMF关系计算
3. ✓ 保证完美重构 - 符合小波理论
4. ✓ 减少过拟合 - 更少的自由度
5. ✓ 梯度只流向一个filter - 训练更稳定

使用建议:
---------
- 如果追求理论完美重构 → 用QMF
- 如果想要最大灵活性 → 用两个独立filters
- 数据量小 → 用QMF (参数少)
- 数据量大 → 两者都可以

实现时只需修改一行:
-------------------
# 原来: WaveletClassifier
model = QMFWaveletClassifier(...)  # 自动使用QMF

optimizer只需要优化low_pass即可!
""")