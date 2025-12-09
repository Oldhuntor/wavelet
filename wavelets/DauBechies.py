import numpy as np
import matplotlib.pyplot as plt
import pywt

# -----------------------------
# 构造测试信号：固定频率，不同相位 (Fs = 500 Hz)
# -----------------------------
fs = 500  # 采样率 Hz
t = np.arange(0, 2, 1 / fs)  # 时间向量 (2秒)
N = len(t)

idx1 = slice(0, N // 3)
idx2 = slice(N // 3, 2 * N // 3)
idx3 = slice(2 * N // 3, N)

f1, f2, f3 = 10, 25, 40  # 信号成分频率 (10, 25, 40 Hz)

segment1 = (
        np.sin(2 * np.pi * f1 * t[idx1]) +
        np.sin(2 * np.pi * f2 * t[idx1] + np.pi / 6) +
        np.sin(2 * np.pi * f3 * t[idx1] + np.pi / 4)
)

segment2 = (
        np.sin(2 * np.pi * f1 * t[idx2]) +
        np.sin(2 * np.pi * f2 * t[idx2]) +
        np.sin(2 * np.pi * f3 * t[idx2])
)

segment3 = (
        np.sin(2 * np.pi * f1 * t[idx3]) +
        np.sin(2 * np.pi * f2 * t[idx3] + np.pi / 6) +
        np.sin(2 * np.pi * f3 * t[idx3] + np.pi / 4)
)

x = np.concatenate([segment1, segment2, segment3])

# -----------------------------
# 离散小波变换 (Daubechies - Db4)
# -----------------------------
wavelet_name = 'db4'
level = 3
coefficients = pywt.wavedec(x, wavelet_name, level=level)
num_levels = len(coefficients)


# --- 最终修正：使用 idwt 逐级重构细节信号 ---

def reconstruct_approximation(coeffs, wavelet_name, N):
    """单独重构近似信号 A_L (将 cD_i 全部设为 None)"""
    # 列表：[cA_L, None, None, ...]
    reconstruction_list = [coeffs[0]] + [None] * (num_levels - 1)
    # 使用 waverec 重构 A3 是安全的
    return pywt.waverec(reconstruction_list, wavelet_name, mode='sym')[:N]


def reconstruct_detail(coeffs, target_index, wavelet_name, N):
    """
    单独重构细节信号 D_i，通过从 D_i 向上逐级使用 idwt 反变换，
    并将近似系数和所有其他细节系数设置为零向量（Zero Coefficients）。
    """
    # 1. 初始化列表：所有系数都设为零向量
    reconstruction_list = []

    # 确保零向量与原始系数长度匹配
    for i in range(num_levels):
        reconstruction_list.append(np.zeros_like(coeffs[i]))

    # 2. 将目标细节系数放入正确位置 (索引 1, 2, 3...)
    # 目标索引是 target_index
    reconstruction_list[target_index] = coeffs[target_index]

    # 3. 逐级进行 idwt 反变换
    # cA_L 是第一个元素 (索引 0)，cD_L 是第二个元素 (索引 1)，等等

    # 从最低频的近似系数 cA_L (索引 0) 和 cD_L (索引 1) 开始
    # 注意：在我们的列表中，cA_L 是索引 0，c D_L 是索引 1。

    # 找到目标所在的最低分解级别 (例如，如果 target_index=3, 意味着 D1)

    # 总是从当前最高的近似系数开始 (当前是 cA_L = 零向量)
    # 目标是在 waverec 中只保留一个 cD_i，并将 cA_L 设置为零。
    # 解决 "shape mismatch" 的最可靠方法是确保所有系数都是数组 (零系数)，而不是 None。

    # 如果 target_index > 0 (即我们目标是细节系数 D_i)
    # 我们的列表是 [cA3, cD3, cD2, cD1]

    # 将 cA_L 设为零，并将所有 *非目标* 的 cD_i 也设为零。
    for i in range(1, num_levels):
        if i != target_index:
            reconstruction_list[i] = np.zeros_like(coeffs[i])

    # 4. 使用 waverec 进行多级重构 (现在所有元素都是数组，解决了 Shape Mismatch)
    # 重构结果是纯粹的 D_i 信号
    return pywt.waverec(reconstruction_list, wavelet_name, mode='sym')[:N]


# -----------------------------
# 重构信号 (Reconstruction) - 采用修正后的健壮方法
# -----------------------------

# A3 (索引 0)
# 注意：近似信号 A_L 的重构仍使用 [cA_L, None, None, ...]，这是安全的。
cA3_reconstructed = reconstruct_approximation(coefficients, wavelet_name, N)

# D3 (索引 1: 31.25 - 62.5 Hz)
cD3_reconstructed = reconstruct_detail(coefficients, 1, wavelet_name, N)

# D2 (索引 2: 62.5 - 125 Hz)
cD2_reconstructed = reconstruct_detail(coefficients, 2, wavelet_name, N)

# D1 (索引 3: 125 - 250 Hz)
cD1_reconstructed = reconstruct_detail(coefficients, 3, wavelet_name, N)

# -----------------------------
# 绘图：比较原始信号和分解/重构结果
# -----------------------------
plt.figure(figsize=(12, 12))

# 原始信号
plt.subplot(5, 1, 1)
plt.plot(t, x, 'k')
plt.title(f'1. Original Signal (Db4, Level {level}): Frequencies 10, 25, 40 Hz')
plt.ylabel('Amplitude')
plt.grid(True)

# 近似信号 A3 (低频部分: 0 - 31.25 Hz)
plt.subplot(5, 1, 2)
plt.plot(t, cA3_reconstructed, 'b')
plt.title(f'2. Approximation Signal (A3 - Low Frequency: 0 - {fs / (2 ** (level + 1)):.2f} Hz)')
plt.ylabel('Amplitude')
plt.grid(True)

# 细节信号 D3 (中低频部分: 31.25 - 62.5 Hz)
plt.subplot(5, 1, 3)
plt.plot(t, cD3_reconstructed, 'g')
plt.title(f'3. Detail Signal (D3 - Approx {fs / (2 ** (level + 1)):.2f} - {fs / (2 ** level):.2f} Hz)')
plt.ylabel('Amplitude')
plt.grid(True)

# 细节信号 D2 (中高频部分: 62.5 - 125 Hz)
plt.subplot(5, 1, 4)
plt.plot(t, cD2_reconstructed, 'r')
plt.title(f'4. Detail Signal (D2 - Approx {fs / (2 ** level):.2f} - {fs / 4:.2f} Hz)')
plt.ylabel('Amplitude')
plt.grid(True)

# 细节信号 D1 (最高频部分: 125 - 250 Hz)
plt.subplot(5, 1, 5)
plt.plot(t, cD1_reconstructed, 'm')
plt.title(f'5. Detail Signal (D1 - Approx {fs / 4:.2f} - {fs / 2:.2f} Hz)')
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.grid(True)

plt.tight_layout()
plt.show()