import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from scipy.sparse.csgraph import laplacian
from scipy.signal import savgol_filter
from numba import njit

@njit
def generate_golomb_ruler(n: int) -> np.ndarray:
    """
    Generates the first n Golomb rulers using an optimized growing algorithm.
    """
    G = np.zeros(n, dtype=np.int64)
    # D will store whether a difference has been seen. Use a larger initial size.
    D_size = 1024
    D = np.zeros(D_size, dtype=np.bool_)

    G[0] = 0
    current_length = 1

    while current_length < n:
        # Start checking from the next integer after the last element
        m = G[current_length - 1] + 1
        #print(f"Checking m={m} at current_length={current_length}")  # Debug

        while True:
            valid = True
            max_diff = 0

            # Check if current m creates any repeated differences with existing G elements
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
                # Secondary check using a temporary array
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
                #print(f"Added m={m} to G, current_length={current_length}")  # Debug
                break
            else:
                m += 1
                #print(f"Incremented m to {m}")  # Debug

    return G

def generate_symbolic_system(n, base_symbols=32, cluster_size=3, corr_prob=0.99):
    symbols = [chr(65 + i) for i in range(base_symbols)]
    golomb_positions = generate_golomb_ruler(n)
    if not golomb_positions.any() or golomb_positions[0] == 0:  # Check if all zeros or empty
        print("Warning: Golomb positions invalid, returning empty system")
        return [], np.array([], dtype=np.int64)
    system = []
    for i, pos in enumerate(golomb_positions):
        cluster_id = i // cluster_size
        np.random.seed(cluster_id)
        length = np.random.randint(12, 16)
        if i > 0 and np.random.random() < corr_prob:
            prev = list(system[i-1].split('_')[1])
            s = ''.join(prev[:length//2]) + ''.join(np.random.choice(symbols, size=length//2))
        else:
            s = ''.join(np.random.choice(symbols, size=length))
        system.append(f"{pos:04d}_{s}")
    return system, golomb_positions

def compute_mi_matrix(system, fixed_length=16, batch_size=1000):
    n = len(system)
    mi_matrix = np.zeros((n, n))
    for i in range(0, n, batch_size):
        for j in range(i, min(n, i + batch_size)):
            a = list(system[j].split('_')[1].ljust(fixed_length, ' ')[0:fixed_length])
            for k in range(max(i, j), min(n, j + batch_size)):
                b = list(system[k].split('_')[1].ljust(fixed_length, ' ')[0:fixed_length])
                mi = mutual_info_score(a, b)
                mi_matrix[j, k] = mi
                mi_matrix[k, j] = mi
    return mi_matrix

def compute_informational_metrics(mi_matrix, n):
    I_max = np.log2(n * (n - 1) / 2) if n > 1 else 0
    ell_info = 1 / (1 + I_max)
    d_matrix = 1 / (1 + mi_matrix + 1e-10)
    np.fill_diagonal(d_matrix, np.inf)
    # Handle case where d_matrix has no finite values
    finite_mask = d_matrix < np.inf
    if not np.any(finite_mask):
        return ell_info, 0.0, 0.0  # Default values if no finite d_min
    d_min = np.min(d_matrix[finite_mask])
    if d_min > ell_info:
        R_n = (1 / ell_info**2) * (1 - d_min / ell_info)
    else:
        R_n = 0
    diff = 1 / ell_info**2 - 1 / d_matrix**2
    E_n = -np.sum(np.log(np.abs(diff) + 1e-10)) / n
    return ell_info, R_n, E_n

def compute_action(mi_matrix, ell_info):
    d_matrix = 1 / (1 + mi_matrix + 1e-10)
    np.fill_diagonal(d_matrix, 0)
    S_G = np.sum(mi_matrix * d_matrix**2)
    return S_G

def detect_transitions(mi_matrix, n, golomb_positions):
    transitions = []
    eigen_stats = []
    curvature_stats = []
    energy_stats = []
    action_stats = []

    min_system_size = 20
    window_size = max(5, min(50, n // 20))
    eps = 1e-6

    I_max = np.log2(n * (n - 1) / 2) if n > 1 else 0
    ell_info = 1 / (1 + I_max)
    epsilon_1 = 1 + ell_info
    epsilon_2 = 1 + ell_info / 2
    curvature_threshold = -3000

    if n < min_system_size or mi_matrix.size == 0:
        print(f"System too small (n={n}) or mi_matrix empty for transition detection")
        return transitions, curvature_stats, energy_stats, action_stats

    try:
        for k in range(window_size, n + 1, max(1, window_size // 2)):
            submatrix = mi_matrix[:k, :k]
            if submatrix.size == 0:
                print(f"Skipping k={k} due to empty submatrix")
                continue
            L = laplacian(submatrix, normed=True)
            eigs = np.sort(np.linalg.eigvalsh(L))[:4]
            print(f"k={k}, eigs={eigs}")

            sub_ell_info, R_n, E_n = compute_informational_metrics(submatrix, k)
            S_G = compute_action(submatrix, sub_ell_info)
            curvature_stats.append((k, R_n))
            energy_stats.append((k, E_n))
            action_stats.append((k, S_G))

            λ1 = max(abs(eigs[0]), eps)
            λ2 = eigs[1]
            λ3 = eigs[2]
            if λ2 > eps:
                ratio12 = λ2 / λ1
                ratio23 = λ3 / λ2
                eigen_stats.append((k, ratio12, ratio23))

    except Exception as e:
        print("Matrix analysis failed:", e)
        return [], [], [], []

    if not eigen_stats:
        print("No valid eigenvalue ratios computed")
        return [], [], [], []

    ks, ratio12, ratio23 = zip(*eigen_stats)

    if len(ks) > 11:
        window_len = min(5, len(ks) // 2 * 2 + 1)
        ratio12_smooth = savgol_filter(ratio12, window_length=window_len, polyorder=2)
        ratio23_smooth = savgol_filter(ratio23, window_length=window_len, polyorder=2)
    else:
        ratio12_smooth = ratio12
        ratio23_smooth = ratio23

    try:
        t1_idx = np.argmax(np.array(ratio12_smooth) > epsilon_1)
        t2_idx = np.argmax(np.array(ratio23_smooth) > epsilon_2)

        min_trans = max(50, n // 25)
        if ks[t1_idx] > min_trans and R_n > curvature_threshold and S_G > 0:
            transitions.append(("1D→2D", ks[t1_idx]))
        if '1D→2D' in [t[0] for t in transitions] and ks[t2_idx] > 1.2 * ks[t1_idx] and R_n > curvature_threshold and S_G > 0:
            transitions.append(("2D→3D", ks[t2_idx]))

    except Exception as e:
        print("Transition analysis failed:", e)

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    plt.plot(ks, ratio12, 'b-', alpha=0.3, label="Raw λ₂/λ₁")
    plt.plot(ks, ratio12_smooth, 'r-', label="Smoothed λ₂/λ₁")
    plt.axhline(epsilon_1, color='k', linestyle='--', label=f"Threshold ({epsilon_1:.3f})")
    plt.xlabel("n distinctions")
    plt.ylabel("λ₂ / λ₁")
    plt.title("1D → 2D Transition")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(ks, ratio23, 'b-', alpha=0.3, label="Raw λ₃/λ₂")
    plt.plot(ks, ratio23_smooth, 'r-', label="Smoothed λ₃/λ₂")
    plt.axhline(epsilon_2, color='k', linestyle='--', label=f"Threshold ({epsilon_2:.3f})")
    plt.xlabel("n distinctions")
    plt.ylabel("λ₃ / λ₂")
    plt.title("2D → 3D Transition")
    plt.legend()
    plt.grid(True)

    ks_c, R_n_values = zip(*curvature_stats)
    plt.subplot(2, 2, 3)
    plt.plot(ks_c, R_n_values, 'g-', label="Informational Curvature R_n")
    plt.axhline(curvature_threshold, color='k', linestyle='--', label=f"Threshold ({curvature_threshold})")
    plt.xlabel("n distinctions")
    plt.ylabel("R_n")
    plt.title("Informational Curvature")
    plt.legend()
    plt.grid(True)

    ks_e, E_n_values = zip(*energy_stats)
    ks_a, S_G_values = zip(*action_stats)
    plt.subplot(2, 2, 4)
    plt.plot(ks_e, E_n_values, 'm-', label="Informational Energy E_n")
    plt.plot(ks_a, S_G_values, 'c-', label="Action S_G")
    plt.axhline(0, color='k', linestyle='--', label="Threshold (0)")
    plt.xlabel("n distinctions")
    plt.ylabel("E_n / S_G")
    plt.title("Informational Energy and Action")
    plt.legend()
    plt.grid(True)
																		
    plt.tight_layout()
    plt.savefig("golomb_transitions_optimized.png")
    print("Plot saved: golomb_transitions_optimized.png")

    print("\nAction Values:", list(zip(ks_a, S_G_values)))

    return transitions, curvature_stats, energy_stats, action_stats

if __name__ == "__main__":
    n = 5000
    system, golomb_positions = generate_symbolic_system(n)
    print("System length:", len(system))  # Debug
    print("Golomb positions:", golomb_positions)  # Debug
    mi_matrix = compute_mi_matrix(system)
    print("MI Matrix shape:", mi_matrix.shape)  # Debug
    transitions, curvature_stats, energy_stats, action_stats = detect_transitions(mi_matrix, n, golomb_positions)

    print("\nDetected Transitions:")
    for label, k in transitions:
        print(f"  {label} at n = {k}")

    D = set(abs(golomb_positions[i] - golomb_positions[j])
            for i in range(len(golomb_positions)) for j in range(i + 1, len(golomb_positions)))
    is_golomb = len(D) == len(golomb_positions) * (len(golomb_positions) - 1) // 2
    print(f"\nGolomb Ruler Property: {'Valid' if is_golomb else 'Invalid'}")

    ell_info, R_n, E_n = compute_informational_metrics(mi_matrix, n)
    print(f"\nInformational Metrics:")
    print(f"  ℓ_info: {ell_info:.6f}")
    print(f"  R_n: {R_n:.6f}")
    print(f"  E_n: {E_n:.6f}")
