import numpy as np
import matplotlib.pyplot as plt

# Create signal
N = 512
t = np.linspace(0, 1, N, endpoint=False)
f1, f2 = 20, 25
signal = 4 * np.cos(2 * np.pi * f1 * t + np.pi / 6) + 1.5 * np.sin(2 * np.pi * f2 * t + np.pi / 3)

# Frequencies to test
freqs = np.arange(0, 50, 0.1)

# Convolution-based Fourier transform
Re_vals = []
Im_vals = []

for f in freqs:
    cos_kernel = np.cos(2 * np.pi * f * t)
    sin_kernel = np.sin(2 * np.pi * f * t)

    Re = np.sum(signal * cos_kernel)
    Im = -np.sum(signal * sin_kernel)

    Re_vals.append(Re)
    Im_vals.append(Im)

Re_vals = np.array(Re_vals)
Im_vals = np.array(Im_vals)
magnitude_conv = np.sqrt(Re_vals ** 2 + Im_vals ** 2)

# True FFT
fft_vals = np.fft.fft(signal)
fft_mag = np.abs(fft_vals)[:50]

# Plot comparison
plt.figure(figsize=(8, 4))
plt.plot(freqs, magnitude_conv, label="Conv-based Magnitude")
# plt.plot(freqs, fft_mag, label="FFT Magnitude")
plt.legend()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Fourier Magnitude Comparison")
plt.grid(True)
plt.show()
