import pywt
import numpy as np


def calculate_shannon_entropy(coeffs):
    """Calculates the normalized Shannon Entropy of the coefficients."""
    # Convert list of coefficient arrays into a single array
    # We use only the detail coefficients typically, but using all is safer for general comparison.
    flattened_coeffs = np.concatenate(coeffs)

    # Square the coefficients (energy)
    energy_coeffs = flattened_coeffs ** 2

    # Calculate probability distribution (normalized energy)
    total_energy = np.sum(energy_coeffs)
    if total_energy == 0:
        return 0.0

    probabilities = energy_coeffs / total_energy

    # Calculate Shannon Entropy: E = - sum(p * log(p))
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-12
    entropy = -np.sum(probabilities * np.log(probabilities + epsilon))
    return entropy


def find_best_wavelet_by_entropy(signal, wavelet_list, level=4):
    """
    Finds the mother wavelet that minimizes the Shannon Entropy of the DWT coefficients.

    Args:
        signal (np.array): The input signal (e.g., your Random Walk data).
        wavelet_list (list): A list of pywt wavelet names (strings).
        level (int): The decomposition level.

    Returns:
        str: The name of the best-suited wavelet.
    """
    best_wavelet = None
    min_entropy = float('inf')

    print("--- Wavelet Selection Results ---")

    for wavelet_name in wavelet_list:
        try:
            # 1. Perform Multilevel Decomposition
            # wavedec returns [cA_L, cD_L, cD_(L-1), ..., cD_1]
            coeffs = pywt.wavedec(signal, wavelet_name, level=level)

            # 2. Calculate Entropy
            entropy = calculate_shannon_entropy(coeffs)

            print(f"Wavelet: {wavelet_name:<10} | Entropy: {entropy:.4f}")

            # 3. Check for Minimum Entropy
            if entropy < min_entropy:
                min_entropy = entropy
                best_wavelet = wavelet_name

        except ValueError:
            # Skip wavelets that cannot handle the signal length at the given level
            print(f"Wavelet: {wavelet_name:<10} | Status: Skipped (Filter length too long)")
            continue

    print(f"\n-> BEST SUITED WAVELET: {best_wavelet} (Min Entropy: {min_entropy:.4f})")
    return best_wavelet


# --- Example Usage with the Random Walk Signal ---
N = 1024
# (Assuming generate_random_signal function from previous turns is defined)
# signal, t = generate_random_signal(N, drift=0.05, step_std_dev=1.5)
# --- For demonstration, creating a simple signal here ---
t = np.linspace(0, 1, N)
signal = np.cos(2 * np.pi * 20 * t) + np.random.randn(N) * 0.5 + 2 * t  # Simple periodic + noise + trend

candidate_wavelets = ['haar', 'db2', 'db4', 'db8', 'sym5', 'coif1', 'bior3.3']

best_filter = find_best_wavelet_by_entropy(signal, candidate_wavelets, level=5)