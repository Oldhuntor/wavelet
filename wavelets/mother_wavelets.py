import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def morlet_wavelet(t, w0=5):
    """
    生成 Morlet 小波（复数形式）
    :param t: 时间张量
    :param w0: 中心频率参数
    """
    return torch.exp(1j * w0 * t) * torch.exp(-t**2 / 2)

# 创建时间轴
t = torch.linspace(-5, 5, steps=400)
psi = morlet_wavelet(t, w0=6)

# 绘制实部和虚部
plt.figure(figsize=(10,4))
plt.plot(t.numpy(), psi.real.numpy(), label='Real part')
plt.plot(t.numpy(), psi.imag.numpy(), label='Imag part')
plt.title('Morlet Wavelet (w0=6)')
plt.legend()
plt.grid(True)
plt.show()