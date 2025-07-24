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

def analyze_growth(n):
    G = golomb_grow(n)

    differences = set()
    diff_counts = []
    for i in range(1, n):
        new_diffs = set()
        for j in range(i):
            diff = G[i] - G[j]
            new_diffs.add(diff)
        differences.update(new_diffs)
        diff_counts.append(len(differences))
																   
    increments = np.diff(G)
    action = [i * (i - 1) // 2 for i in range(1, n + 1)]

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(range(1, n), diff_counts, label="Distinct Pairwise Differences")
    axs[0].set_ylabel("Total Unique Differences")
    axs[0].set_title("Growth of Distinctions")

    axs[1].plot(increments, label="Δm = G[n+1] - G[n]")
    axs[1].set_ylabel("Increment Size")
    axs[1].set_title("Combinatorial Tension")

    axs[2].plot(action, label="Redundant Action (ℰ)")
    axs[2].set_ylabel("ℰ(n) = ∑ Unique Distinctions")
    axs[2].set_xlabel("n (steps)")
    axs[2].set_title("Least Redundant Action Curve")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

# Run it
analyze_growth(100)
