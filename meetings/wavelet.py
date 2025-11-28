import numpy as np
import matplotlib.pyplot as plt
import pywt

# 构造信号：左边高频，右边低频
t = np.linspace(0, 1, 500)
x = np.concatenate([
    np.sin(2 * np.pi * 50 * t[:250]),
    np.sin(2 * np.pi * 5 * t[250:])
])

# 定义尺度范围
widths = np.arange(1, 31)

# 连续小波变换 (CWT)
coeffs, freqs = pywt.cwt(x, widths, 'morl')

# 绘制结果
plt.figure(figsize=(10,6))
plt.imshow(np.abs(coeffs), extent=[0,1,widths.min(),widths.max()],
           cmap='magma', aspect='auto', origin='lower')
plt.colorbar(label='|CWT coefficients|')
plt.xlabel('Time')
plt.ylabel('Scale (related to frequency)')
plt.title('Scalogram using PyWavelets')
plt.show()