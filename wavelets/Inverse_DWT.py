import matplotlib.pyplot as plt
import numpy as np


def dwt(signal, low_pass, high_pass):
    """Discrete Wavelet Transform (single level)

    Args:
        signal: Input signal (1D array)
        low_pass: Low-pass filter coefficients
        high_pass: High-pass filter coefficients

    Returns:
        (approximation, detail) coefficients
    """
    signal = np.asarray(signal)
    low_pass = np.asarray(low_pass)
    high_pass = np.asarray(high_pass)

    # Convolve and downsample
    approx = np.convolve(signal, low_pass, mode='full')[::2]
    detail = np.convolve(signal, high_pass, mode='full')[::2]

    return approx, detail


def dwt_multilevel(signal, low_pass, high_pass, levels):
    """Multi-level Discrete Wavelet Transform

    Args:
        signal: Input signal (1D array)
        low_pass: Low-pass filter coefficients
        high_pass: High-pass filter coefficients
        levels: Number of decomposition levels

    Returns:
        [approx_N, detail_N, detail_N-1, ..., detail_1]
    """
    coeffs = []
    current = signal

    for _ in range(levels):
        approx, detail = dwt(current, low_pass, high_pass)
        coeffs.append(detail)
        current = approx

    coeffs.append(current)  # Final approximation
    return coeffs[::-1]  # Reverse to [approx_N, detail_N, ..., detail_1]


def idwt(approx, detail, low_pass, high_pass, signal_len=None):
    """Inverse Discrete Wavelet Transform (single level)

    Args:
        approx: Approximation coefficients
        detail: Detail coefficients
        low_pass: Low-pass reconstruction filter
        high_pass: High-pass reconstruction filter
        signal_len: Original signal length (if None, uses len(approx)*2)

    Returns:
        Reconstructed signal
    """
    approx = np.asarray(approx)
    detail = np.asarray(detail)
    low_pass = np.asarray(low_pass)
    high_pass = np.asarray(high_pass)

    if signal_len is None:
        signal_len = len(approx) * 2

    # Use max length for upsampling
    max_len = max(len(approx), len(detail))

    # Upsample by inserting zeros
    up_approx = np.zeros(max_len * 2)
    up_approx[::2][:len(approx)] = approx

    up_detail = np.zeros(max_len * 2)
    up_detail[::2][:len(detail)] = detail

    # Convolve with reconstruction filters
    rec_approx = np.convolve(up_approx, low_pass, mode='full')
    rec_detail = np.convolve(up_detail, high_pass, mode='full')

    # Trim to remove boundary effects
    start = len(low_pass) - 1
    min_len = min(len(rec_approx), len(rec_detail))
    return (rec_approx[:min_len] + rec_detail[:min_len])[start:start + signal_len]


def idwt_multilevel(coeffs, low_pass, high_pass, signal_len):
    """Multi-level Inverse Discrete Wavelet Transform

    Args:
        coeffs: [approx_N, detail_N, detail_N-1, ..., detail_1]
        low_pass: Low-pass reconstruction filter
        high_pass: High-pass reconstruction filter
        signal_len: Original signal length

    Returns:
        Reconstructed signal
    """
    levels = len(coeffs) - 1
    current = coeffs[0]

    for i in range(levels):
        detail = coeffs[i + 1]
        # Each level doubles the length
        target_len = min(len(current) * 2, signal_len)
        current = idwt(current, detail, low_pass, high_pass, target_len)

    return current[:signal_len]


def generate_signal(frequencies, duration=1.0, sampling_rate=1000):
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    signal = sum(np.sin(2 * np.pi * f * t) for f in frequencies)
    return t, signal



# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t, signal = generate_signal([5, 15, 30])

    plt.plot(signal)
    plt.show()
    # Haar wavelet filters
    low_pass = np.array([1, 1]) / np.sqrt(2)
    high_pass = np.array([1, -1]) / np.sqrt(2)

    print("=== Single-level DWT ===")
    approx, detail = dwt(signal, low_pass, high_pass)
    reconstructed = idwt(approx, detail, low_pass[::-1], high_pass[::-1], len(signal))
    print(f"Error: {np.max(np.abs(signal - reconstructed)):.2e}")

    print("\n=== 3-level DWT ===")
    coeffs = dwt_multilevel(signal, low_pass, high_pass, levels=3)
    for level in range(4):
        plt.plot(coeffs[level], label=f"Level {4-level}")
        plt.legend()
        plt.show()

    print(f"Approx (level 3): {coeffs[0]}")
    print(f"Detail (level 3): {coeffs[1]}")
    print(f"Detail (level 2): {coeffs[2]}")
    print(f"Detail (level 1): {coeffs[3]}")

    reconstructed = idwt_multilevel(coeffs, low_pass[::-1], high_pass[::-1], len(signal))
    print(f"\nReconstructed: {reconstructed}")
    print(f"Error: {np.max(np.abs(signal - reconstructed)):.2e}")