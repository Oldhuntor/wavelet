import numpy as np
import matplotlib.pyplot as plt
import pywt

fs = 200                     # 采样率 Hz
t = np.arange(0, 2, 1/fs)     # 时间向量 (2秒)

N = len(t)
# 分段索引
idx1 = slice(0, N//3)
idx2 = slice(N//3, 2*N//3)
idx3 = slice(2*N//3, N)

# 固定频率成分
f1, f2, f3 = 10, 25, 40       # 三个频率成分

# --- 第一段：初始相位 ---
segment1 = (
    np.sin(2*np.pi*f1*t[idx1]) +
    np.sin(2*np.pi*f2*t[idx1] + np.pi/6) +
    np.sin(2*np.pi*f3*t[idx1] + np.pi/4)
)

# --- 第二段：改变每个成分的相位 ---
segment2 = (
    np.sin(2*np.pi*f1*t[idx2] + np.pi/3) +
    np.sin(2*np.pi*f2*t[idx2] + np.pi/2) +
    np.sin(2*np.pi*f3*t[idx2] + np.pi)
)

# --- 第三段：再换一组不同相位 ---
segment3 = (
    np.sin(2*np.pi*f1*t[idx3] - np.pi/8) +
    np.sin(2*np.pi*f2*t[idx3] - np.pi/4) +
    np.sin(2*np.pi*f3*t[idx3] - np.pi/6)
)

# 拼接成完整信号
x = np.concatenate([segment1, segment2, segment3])

# 2. 连续小波变换 (CWT)
scales = np.arange(1, 128)    # 尺度范围
coefficients, frequencies = pywt.cwt(x, scales, 'cmor1.5-1.0', sampling_period=500)
# coefficients 是复数矩阵，每个元素包含幅度和相位信息
amplitude = np.abs(coefficients)
phase = np.angle(coefficients)

# 3. 绘图部分
plt.figure(figsize=(12,10))

# --- 原始信号 ---
plt.subplot(3,1,1)
plt.plot(t, x, color='black')
plt.title('Original Signal')
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.grid(True)

# --- 幅度谱（时频图） ---
plt.subplot(3,1,2)
plt.imshow(amplitude,
           extent=[t[0], t[-1], frequencies[-1], frequencies[0]],
           cmap='jet', aspect='auto')
plt.title('Morlet CWT Amplitude Spectrum')
plt.ylabel('Frequency [Hz]')
plt.colorbar(label='Amplitude')

# --- 相位谱 ---
plt.subplot(3,1,3)
plt.imshow(phase,
           extent=[t[0], t[-1], frequencies[-1], frequencies[0]],
           cmap='twilight', aspect='auto')
plt.title('Morlet CWT Phase Spectrum')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar(label='Phase [radians]')

plt.tight_layout()
plt.show()