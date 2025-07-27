import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from mpl_toolkits.mplot3d import Axes3D
import os

# Generate greedy Golomb ruler sequence
def generate_golomb_sequence(n):
    G = [0]
    D = set()
    while len(G) < n:
        m = G[-1] + 1
        while True:
            diffs = {abs(m - g) for g in G}
            if D.isdisjoint(diffs):
                break
            m += 1
        G.append(m)
        D.update(abs(m - g) for g in G[:-1])
    return np.array(G)

# Simulate distinction data with stronger correlation
def simulate_distinction_data(G, length=500):
    np.random.seed(42) 
    base = np.random.randint(0, 2, length)                        
    return {x: base ^ (np.random.random(length) < 0.02).astype(int) for x in G}

# Compute mutual information (simplified for speed)
def mutual_information(x, y):
    joint_prob = np.histogram2d(x, y, bins=2)[0] / len(x)
    joint_prob = joint_prob / joint_prob.sum()
    nz = joint_prob > 0
    return np.sum(joint_prob[nz] * np.log(joint_prob[nz] / (joint_prob.sum(axis=1)[:, None] * joint_prob.sum(axis=0)[None, :])[nz]))

# Build mutual information matrix
def compute_mi_matrix(data):
    keys = list(data.keys())
    n = len(keys)
    mi_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mi = mutual_information(data[keys[i]], data[keys[j]])
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    print(f"MI matrix max: {np.max(mi_matrix):.6f}, min (off-diag): {np.min(mi_matrix[np.triu_indices(n, k=1)]):.6f}")
    return mi_matrix, keys

# Compute informational metrics with tighter percentile threshold
def compute_informational_metrics(mi_matrix, n):
    I_max = np.log(n * (n - 1) / 2) if n > 1 else 0
    ell_info = 1 / (1 + I_max)
    d_matrix = 1 / (1 + mi_matrix + 1e-10)
    np.fill_diagonal(d_matrix, np.inf)
    threshold = np.percentile(mi_matrix[np.triu_indices(n, k=1)], 1)  # 1st percentile
    d_matrix[mi_matrix < threshold] = np.inf
    finite_dists = d_matrix[d_matrix < np.inf]
    d_min = np.min(finite_dists) if len(finite_dists) > 0 else ell_info
    R_n = (1 / ell_info**2) * (1 - d_min / ell_info) if d_min > ell_info else 0
    return ell_info, R_n, d_min

# Dimensional bifurcation analysis
def analyze_dimensionality(mi_matrix, n, eps1=1.5, eps2=1.2):
    D = np.diag(np.sum(mi_matrix, axis=1))
    L = D - mi_matrix
    eigenvalues = eigh(L, subset_by_index=[0, min(3, n-1)])[0]
    return eigenvalues[1:4] if len(eigenvalues) > 3 else eigenvalues[1:]
					   
# Plot MI landscape
def plot_mi_landscape(mi_matrix, title="MI Landscape"):
    n = mi_matrix.shape[0]
    x, y = np.meshgrid(range(n), range(n))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, mi_matrix, cmap='terrain', edgecolor='none', alpha=0.8)
    ax.set_xlabel('Golomb Position Index')
    ax.set_ylabel('Golomb Position Index')
    ax.set_zlabel('Mutual Information')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5, label='MI Value')
    plt.tight_layout()
    plt.savefig(f'/Users/ralf/Downloads/{title.replace(" ", "_")}.png')  # Save to specific path
    plt.show()

# Main simulation with trend plotting
def main():
    n_values = [50, 100]
    ratios12, ratios23 = [], []
    
    for n in n_values:
        print(f"\nSimulating with n={n} distinctions...")
        G = generate_golomb_sequence(n)
        print(f"Golomb positions: {G[:10]}... (total {len(G)})")
        
        data = simulate_distinction_data(G)
        mi_matrix, keys = compute_mi_matrix(data)
        print("MI matrix computed.")
        
        plot_mi_landscape(mi_matrix, title=f"MI Landscape (n={n})")
        
        ell_info, R_n, d_min = compute_informational_metrics(mi_matrix, n)
        print(f"ell_info: {ell_info:.6f}, R_n: {R_n:.6f}, d_min: {d_min:.6f}")
        
        eigenvalues = analyze_dimensionality(mi_matrix, n)
        ratio12 = eigenvalues[1] / eigenvalues[0] if len(eigenvalues) > 1 else 0
        ratio23 = eigenvalues[2] / eigenvalues[1] if len(eigenvalues) > 2 else 0
        ratios12.append(ratio12)
        ratios23.append(ratio23)
        print(f"Ratio λ2/λ1: {ratio12:.3f}, Ratio λ3/λ2: {ratio23:.3f}")
        if ratio12 > 1.5:
            print("Transition to 2D detected.")
        if ratio23 > 1.2 and ratio12 > 1.5:
            print("Transition to 3D detected.")
    
    # Plot eigenvalue ratios
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, ratios12, label="λ2/λ1")
    plt.plot(n_values, ratios23, label="λ3/λ2")
    plt.axhline(1.5, color='r', linestyle='--', label="2D Threshold")
    plt.axhline(1.2, color='g', linestyle='--', label="3D Threshold")
    plt.xlabel("Number of Distinctions (n)")
    plt.ylabel("Eigenvalue Ratio")
    plt.title("Dimensional Bifurcation Trends")
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/ralf/Downloads/bifurcation_trends.png')  # Save to specific path
    plt.show()

if __name__ == "__main__":
    main()
