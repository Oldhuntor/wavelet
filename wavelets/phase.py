import numpy as np
import matplotlib.pyplot as plt
import pywt

# -----------------------------
# 构造测试信号：固定频率，不同相位
# -----------------------------
fs = 500                      # 采样率 Hz
t = np.arange(0, 2, 1/fs)     # 时间向量 (2秒)
N = len(t)

idx1 = slice(0, N//3)
idx2 = slice(N//3, 2*N//3)
idx3 = slice(2*N//3, N)

f1, f2, f3 = 10, 25, 40       # 固定三个频率成分

segment1 = (
    np.sin(2*np.pi*f1*t[idx1]) +
    np.sin(2*np.pi*f2*t[idx1] + np.pi/6) +
    np.sin(2*np.pi*f3*t[idx1] + np.pi/4)
)

segment2 = (
    np.sin(2*np.pi*f1*t[idx2]) +
    np.sin(2*np.pi*f2*t[idx2] ) +
    np.sin(2*np.pi*f3*t[idx2])
)

segment3 = (
    np.sin(2*np.pi*f1*t[idx3]) +
    np.sin(2*np.pi*f2*t[idx3] + np.pi/6) +
    np.sin(2*np.pi*f3*t[idx3] + np.pi/4)
)

x = np.concatenate([segment1, segment2, segment3])

# -----------------------------
# 连续小波变换 (Morlet)
# -----------------------------
scales = np.arange(1, 128)
coefficients, frequencies = pywt.cwt(x, scales, 'cmor1.5-1.0', sampling_period=1/1000000)

amplitude = np.abs(coefficients)
phase = np.angle(coefficients)

# -----------------------------
# 绘图比较结果
# -----------------------------
plt.figure(figsize=(12,10))

# 原始信号
plt.subplot(3,1,1)
plt.plot(t,x,'k')
plt.title('Original Signal: Same Frequencies but Different Phases')
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.grid(True)

# 幅度谱（时频图）
plt.subplot(3,1,2)
plt.imshow(amplitude,
           extent=[t[0], t[-1], frequencies[-1], frequencies[0]],
           cmap='jet', aspect='auto')
plt.title('Morlet CWT Amplitude Spectrum')
plt.ylabel('Frequency [Hz]')
plt.colorbar(label='Amplitude')

# 相位谱（时频图）
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