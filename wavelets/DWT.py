import torch
import torch.nn.functional as F

def morlet_wavelet(t, w0=5):
    return torch.exp(1j * w0 * t) * torch.exp(-t**2 / 2)

# 离散尺度集合
scales = [1., 2., 4., 8.]
t = torch.linspace(-5, 5, steps=400)
signal = torch.sin(2*torch.pi*7*t).unsqueeze(0).unsqueeze(0)

coeffs = []
for s in scales:
    psi = morlet_wavelet(t/s)
    kernel_real = psi.real.unsqueeze(0).unsqueeze(0)
    conv_real = F.conv1d(signal, kernel_real, padding=t.numel()//2)
    coeffs.append(conv_real.squeeze().detach().numpy())