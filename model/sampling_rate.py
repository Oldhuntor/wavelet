import pywt
import numpy as np

def morlet_kernel(
    kernel_size: int,
    scale: float = 1.0,
    normalize: bool = True
):
    """
    Returns Morlet wavelet coefficients suitable for CNN1d initialization.
    """

    wavelet = pywt.ContinuousWavelet('morl')

    # sample the mother wavelet
    psi, x = wavelet.wavefun(length=kernel_size)

    # apply scale (time stretching)
    if scale != 1.0:
        new_len = int(kernel_size * scale)
        psi = np.interp(
            np.linspace(0, kernel_size - 1, new_len),
            np.arange(kernel_size),
            psi
        )

    # energy normalization (recommended)
    if normalize:
        psi = psi / np.linalg.norm(psi)

    return psi.astype(np.float32)

kernel_size = 31      # MUST be odd
scale = 5             # controls frequency

kernel = morlet_kernel(kernel_size, scale)

print(kernel.shape)
print(kernel)
