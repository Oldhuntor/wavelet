"""
离散小波变换 (DWT) 的小波数量
================================

标准DWT: 2个小波
----------------
- Low-pass filter (approximation) - 低频成分
- High-pass filter (detail) - 高频成分

这是最常见的形式，称为"二进小波分解" (dyadic wavelet decomposition)


多小波DWT: N个小波
------------------
可以将信号分解成多个频带，而不只是低频和高频。

常见的多小波分解:
1. **Wavelet Packet Decomposition (小波包分解)**
   - 不只分解approximation，也分解detail
   - 可以得到更精细的频率划分

2. **M-band Wavelet (M带小波)**
   - 直接使用M个不同的filters
   - 每个filter捕获不同的频带
   - 例如: 3-band, 4-band wavelet

3. **Filter Bank (滤波器组)**
   - 使用多个bandpass filters
   - 类似STFT但用wavelet


为什么要用多小波?
-----------------
1. 更精细的频率分辨率
2. 捕获更多频域特征
3. 适合某些特定信号(如音频、生物信号)
4. 可学习的多尺度表示


实现示例
--------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiWaveletTransform(nn.Module):
    def __init__(self, filter_length, levels, num_wavelets=2, init_type='random',
                 use_frequency_constraint=True):
        """多小波变换

        Args:
            num_wavelets: 小波数量 (默认2: low-pass + high-pass)
                         可以设置为3, 4, 8等
        """
        super().__init__()
        self.filter_length = filter_length
        self.levels = levels
        self.num_wavelets = num_wavelets
        self.use_frequency_constraint = use_frequency_constraint

        # 创建多个可学习的filters
        if init_type == 'random':
            self.filters = nn.ParameterList([
                nn.Parameter(torch.randn(filter_length) / np.sqrt(filter_length))
                for _ in range(num_wavelets)
            ])
        elif init_type == 'frequency_init':
            # 按频率范围初始化
            self.filters = nn.ParameterList([
                self._init_frequency_filter(i, filter_length)
                for i in range(num_wavelets)
            ])

        # Normalize
        with torch.no_grad():
            for f in self.filters:
                f.data = F.normalize(f.data, dim=0)

    def _init_frequency_filter(self, index, length):
        """初始化特定频率的filter"""
        # 为每个filter分配不同的频率范围
        freq_range = (index / self.num_wavelets, (index + 1) / self.num_wavelets)

        # 创建对应频段的filter
        t = torch.linspace(0, 1, length)
        center_freq = (freq_range[0] + freq_range[1]) / 2

        # 使用调制的高斯函数
        filter_data = torch.cos(2 * np.pi * center_freq * 10 * t) * \
                      torch.exp(-((t - 0.5) ** 2) / 0.1)

        return nn.Parameter(filter_data)

    def apply_frequency_constraint(self, filter_param, target_band_index):
        """为每个filter应用频率约束"""
        if not self.use_frequency_constraint:
            return F.normalize(filter_param, dim=0)

        # 计算目标频带
        freq_start = target_band_index / self.num_wavelets
        freq_end = (target_band_index + 1) / self.num_wavelets

        # FFT
        fft_size = max(64, self.filter_length * 4)
        filter_padded = F.pad(filter_param, (0, fft_size - self.filter_length))
        filter_fft = torch.fft.rfft(filter_padded)
        magnitude = torch.abs(filter_fft)
        phase = torch.angle(filter_fft)

        # 创建频带权重 (只保留目标频带)
        freqs = torch.linspace(0, 1, len(magnitude), device=filter_param.device)

        # 高斯窗口在目标频带
        center = (freq_start + freq_end) / 2
        width = (freq_end - freq_start) / 2
        weights = torch.exp(-((freqs - center) / width) ** 2)

        # 应用约束
        constrained_magnitude = magnitude * weights
        constrained_fft = constrained_magnitude * torch.exp(1j * phase)
        constrained_filter = torch.fft.irfft(constrained_fft, n=fft_size)
        constrained_filter = constrained_filter[:self.filter_length]

        return F.normalize(constrained_filter, dim=0)

    def get_constrained_filters(self):
        """获取约束后的filters"""
        constrained = []
        for i, f in enumerate(self.filters):
            constrained.append(
                self.apply_frequency_constraint(f, i)
            )
        return constrained

    def dwt_single_level(self, signal):
        """单层多小波分解

        Returns:
            List of coefficient tensors, one per wavelet
        """
        signal = signal.unsqueeze(1)  # (batch, 1, length)

        filters = self.get_constrained_filters()
        # filters = self.filters
        coeffs = []

        for f in filters:
            f_kernel = f.view(1, 1, -1)
            conv_result = F.conv1d(signal, f_kernel, padding=self.filter_length-1)
            downsampled = conv_result[:, :, ::2]  # Downsample
            coeffs.append(downsampled.squeeze(1))

        return coeffs

    def dwt_multilevel(self, signal):
        """多层多小波分解

        Returns:
            List of lists: [[level1_w1, level1_w2, ...], [level2_w1, ...], ...]
        """
        all_coeffs = []
        current = signal

        for _ in range(self.levels):
            level_coeffs = self.dwt_single_level(current)
            all_coeffs.append(level_coeffs)
            # 用第一个小波的系数继续分解 (类似approximation)
            current = level_coeffs[0]

        return all_coeffs

    def forward(self, signal):
        """前向传播"""
        return self.dwt_multilevel(signal)


class MultiWaveletClassifier(nn.Module):
    def __init__(self, filter_length, levels, signal_length, num_classes,
                 num_wavelets=2, hidden_dim=128, init_type='random'):
        """多小波分类器

        Args:
            num_wavelets: 使用多少个小波 (2=标准DWT, 3/4/8=多频带)
        """
        super().__init__()
        self.wavelet = MultiWaveletTransform(
            filter_length, levels, num_wavelets, init_type
        )
        self.signal_length = signal_length
        self.num_wavelets = num_wavelets

        # 计算总系数维度
        total_dim = self._calculate_coeff_dim()

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.num_classes = num_classes

    def _calculate_coeff_dim(self):
        """计算系数总维度"""
        total_dim = 0
        temp_len = self.signal_length

        for _ in range(self.wavelet.levels):
            conv_len = temp_len + self.wavelet.filter_length - 1
            temp_len = (conv_len + 1) // 2
            # 每层有 num_wavelets 个系数
            total_dim += temp_len * self.num_wavelets

        return total_dim

    def forward(self, x):
        """前向传播"""
        # 多层系数: [[w1_l1, w2_l1, ...], [w1_l2, ...], ...]
        x = x.squeeze(1)
        multi_level_coeffs = self.wavelet(x)

        # 展平所有系数
        flat_coeffs = []
        for level_coeffs in multi_level_coeffs:
            for coeff in level_coeffs:
                flat_coeffs.append(coeff.flatten(1))

        flat_coeffs = torch.cat(flat_coeffs, dim=1)

        # 分类
        logits = self.classifier(flat_coeffs)

        return logits


# ============================================================================
# 对比实验
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("多小波 vs 标准小波对比")
    print("="*70)

    batch_size = 8
    signal_length = 128
    num_classes = 3

    # 生成测试数据
    signals = torch.randn(batch_size, signal_length)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 测试不同数量的小波
    for num_wavelets in [2, 4, 8]:
        print(f"\n{'='*70}")
        print(f"使用 {num_wavelets} 个小波")
        print(f"{'='*70}")

        model = MultiWaveletClassifier(
            filter_length=8,
            levels=3,
            signal_length=signal_length,
            num_classes=num_classes,
            num_wavelets=num_wavelets,
            hidden_dim=64,
            init_type='frequency_init'
        )

        # 前向传播
        logits, coeffs = model(signals)

        print(f"\n输出:")
        print(f"  Logits shape: {logits.shape}")
        print(f"  层数: {len(coeffs)}")
        print(f"  每层小波数: {num_wavelets}")

        print(f"\n每层系数形状:")
        for i, level_coeffs in enumerate(coeffs):
            shapes = [c.shape for c in level_coeffs]
            print(f"  Level {i+1}: {shapes}")

        # 计算参数
        filter_params = sum(f.numel() for f in model.wavelet.filters)
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
        total_params = filter_params + classifier_params

        print(f"\n参数统计:")
        print(f"  Wavelet filters: {filter_params:,}")
        print(f"  Classifier: {classifier_params:,}")
        print(f"  Total: {total_params:,}")

        # 测试训练
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        loss.backward()

        print(f"\n训练测试:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  ✓ 梯度计算成功")

    print("\n" + "="*70)
    print("总结")
    print("="*70)
