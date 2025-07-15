import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def golomb_grow(n: int) -> np.ndarray:
    G = np.zeros(n, dtype=np.int64)
    D_size = 1024
    D = np.zeros(D_size, dtype=np.bool_)

    G[0] = 0
    current_length = 1

    while current_length < n:
        m = G[current_length - 1] + 1
                                    
        while True:
            valid = True
            max_diff = 0
                                     
            for i in range(current_length):
                diff = m - G[i]
                if diff >= D_size:
                    new_size = max(D_size * 2, diff + 1)
                    new_D = np.zeros(new_size, dtype=np.bool_)
                    new_D[:D_size] = D
                    D = new_D
                    D_size = new_size

                if D[diff]:
                    valid = False
                    break
                if diff > max_diff:
                    max_diff = diff

            if valid:
                temp = np.zeros(max_diff + 1, dtype=np.bool_)
                for i in range(current_length):
                    diff = m - G[i]
                    if temp[diff]:
                        valid = False
                        break
                    temp[diff] = True

            if valid:
                for i in range(current_length):
                    diff = m - G[i]
                    D[diff] = True
                G[current_length] = m
                current_length += 1
                break
            else:
                m += 1

    return G


@njit
def create_signal_from_golomb(G: np.ndarray) -> np.ndarray:
    signal = np.zeros(G[-1] + 1)
    signal[G] = 1
    signal -= np.mean(signal) # Deactivated to keep DC component
    signal /= np.max(np.abs(signal))
    return signal

def compute_spectrum(signal: np.ndarray):
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(X))
    mag = np.abs(X)
    power = mag ** 2
    return freqs, mag, power

def plot_results(signal: np.ndarray, freqs: np.ndarray, mag: np.ndarray, power: np.ndarray, n: int):
    idx = (freqs > 0) & (freqs <= 0.5)
    freqs = freqs[idx]
    mag = mag[idx]
    power = power[idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    time_indices = np.arange(len(signal))

    # Time-domain stem: plot positives in blue, negatives in red
    pos_idx = signal >= 0
    neg_idx = signal < 0

    # Plot positive values
    axes[0].stem(time_indices[pos_idx], signal[pos_idx], basefmt=" ", linefmt='b-', markerfmt='bo')
    
    # Only plot negative values if there are any
    if np.any(neg_idx): # Check if neg_idx is not empty
        axes[0].stem(time_indices[neg_idx], signal[neg_idx], basefmt=" ", linefmt='r-', markerfmt='ro')
    
    axes[0].set_xscale("log")
    axes[0].set_title(f"Golomb Signal (n = {n}) - Time Domain")
    axes[0].set_xlabel("Time Index")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # FFT Magnitude Spectrum without lines
    axes[1].plot(freqs, mag, color='lightgray')  # optional background curve
    axes[1].stem(freqs, mag, basefmt=" ", linefmt='none', markerfmt='bo')
    #axes[1].set_yscale("log")
    axes[1].set_title("FFT Magnitude Spectrum")
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True)
    axes[1].set_xlim(0, 0.5)

    # Power Spectrum (log scale) without lines
    axes[2].semilogy(freqs, power, color='lightgray')  # optional background curve
    axes[2].stem(freqs, power, basefmt=" ", linefmt='none', markerfmt='go')
    axes[2].set_yscale("log")
    axes[2].set_title("Power Spectrum (log scale)")
    axes[2].set_xlabel("Frequency")
    axes[2].set_ylabel("Power")
    axes[2].grid(True)
    axes[2].set_xlim(0, 0.5)

    plt.tight_layout()
    plt.show()


def main(n: int):
    G = golomb_grow(n)
    signal = create_signal_from_golomb(G)
    freqs, mag, power = compute_spectrum(signal)
    plot_results(signal, freqs, mag, power, n)

# --- Execute ---
if __name__ == "__main__":
    main(n=17)
