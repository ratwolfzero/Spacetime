import numpy as np
from scipy.linalg import eigh
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@njit
def generate_golomb(n: int) -> np.ndarray:
    """
    Generates the first n Golomb rulers using an optimized growing algorithm,
    with preallocated and reused temp arrays for speed.
    """
    G = np.zeros(n, dtype=np.int64)
    D_size = 1024
    D = np.zeros(D_size, dtype=np.bool_)
    
    # Preallocate temp array for internal difference check
    temp_size = 1024
    temp = np.zeros(temp_size, dtype=np.bool_)

    G[0] = 0
    current_length = 1
                                             
    while current_length < n:
        m = G[current_length - 1] + 1

        while True:
            valid = True
            max_diff = 0

            # Check for global difference duplication
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
                # Ensure temp array is large enough
                if max_diff >= temp_size:
                    new_temp_size = max(temp_size * 2, max_diff + 1)
                    new_temp = np.zeros(new_temp_size, dtype=np.bool_)
                    new_temp[:temp_size] = temp
                    temp = new_temp
                    temp_size = new_temp_size

                # Check for internal duplicates among new differences
                temp[:max_diff + 1] = False  # reset only used portion
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

    return G.astype(np.float64)

def compute_metrics(G):
    """Numerically stable metric calculation"""
    n = len(G)
    if n < 2:
        return 1.0, 1.0, 0.0, np.zeros((n,n))
    
    # Safe difference calculation
    diffs = np.abs(np.subtract.outer(G, G))
    np.fill_diagonal(diffs, np.inf)
    mean_diff = np.mean(diffs[diffs < np.inf])
    norm_diffs = diffs / (mean_diff + 1e-16)
    
    # Mutual information matrix                           
    W = np.log(1 + 1/norm_diffs)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
                                                          
    # Framework metrics
    I_max = np.max(W)
    d_min = 1/(1 + I_max)
    l_info = 1/(1 + np.log(n))
    R_n = max(0, (1/l_info) * (1 - d_min/l_info))
    
    return d_min, l_info, R_n, W

def compute_embedding(G, dim):
    """Compute spectral embedding in 2D, 3D, 4D, or 5D using Laplacian eigenvectors"""
    n = len(G)
    if n < 2:
        return np.zeros((n, dim))
    
    # Compute Laplacian
    _, _, _, W = compute_metrics(G)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    try:
        # Get eigenvectors for smallest non-zero eigenvalues (skip λ0)
        _, eigenvectors = eigh(L, eigvals_only=False, subset_by_index=[1, min(dim + 1, n)])
        return eigenvectors
    except Exception as e:
        print(f"Error in compute_embedding for dim={dim}: {e}")
        return np.zeros((n, dim))

def check_transitions(G, d_min, l_info, R_n):
    n = len(G)
    if n < 10:
        return False, False, False, False, 0.0, 0.0, 0.0, 0.0

    try:
        _, _, _, W = compute_metrics(G)
        D = np.diag(np.sum(W, axis=1))
        L = D - W 
                                                               
        eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, 5])
        λ0, λ1, λ2, λ3, λ4, λ5 = eigenvalues

        λ1 = max(λ1, 1e-8)
        λ2 = max(λ2, 1e-8)
        λ3 = max(λ3, 1e-8)
        λ4 = max(λ4, 1e-8)
        λ5 = max(λ5, 1e-8)

        r1 = λ2 / λ1  # For 1D→2D
        r2 = λ3 / λ2  # For 2D→3D
        r3 = λ4 / λ3  # For 3D→4D
        r4 = λ5 / λ4  # For 4D→5D

        transition_2D = (r1 > 1.3 and R_n > 1.3)
        transition_3D = (r2 > 1.15 and R_n > 2.2)
        transition_4D = (r3 > 1.01 and R_n > 3.0)
        transition_5D = (r4 > 1.00 and R_n > 4.0)

        return transition_2D, transition_3D, transition_4D, transition_5D, r1, r2, r3, r4
                                                                                                   
    except Exception as e:
        print("Error in check_transitions:", e)
        return False, False, False, False, 0.0, 0.0, 0.0, 0.0

def validate_golomb(G):
    """Validate that G is a Golomb ruler (all pairwise differences are unique)"""
    n = len(G)
    diffs = np.abs(np.subtract.outer(G, G))
    np.fill_diagonal(diffs, np.inf)
    unique_diffs = np.unique(diffs[diffs < np.inf])
    expected_diffs = n * (n - 1) // 2
    is_valid = len(unique_diffs) == expected_diffs
    entropy = expected_diffs  # S_n = binomial(n, 2)
    return is_valid, entropy

def plot_results(G_full, results, metrics_history):
    """Generate and display graphical outputs for Golomb ruler, mutual information, eigenvalues, transitions, and embeddings"""
    n_max = len(G_full)
    ns, d_mins, l_infos, R_ns, r1s, r2s, r3s, r4s = metrics_history

    # Plot 1: Golomb Ruler Growth
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_max + 1), G_full, 'o-', label='Golomb Ruler Marks')
    plt.xlabel('Index (n)')
    plt.ylabel('Golomb Ruler Value')
    plt.title('Growth of Golomb Ruler')
    plt.grid(True)
    plt.legend()
    plt.savefig('golomb_ruler_growth.png')
    plt.show()
    plt.close()

    # Plot 2: Mutual Information Matrix at n_max
    _, _, _, W = compute_metrics(G_full)
    plt.figure(figsize=(8, 6))
    plt.imshow(W, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Mutual Information I(X_i; X_j)')
    plt.title(f'Mutual Information Matrix at n={n_max}')
    plt.xlabel('Distinction i')
    plt.ylabel('Distinction j')
    plt.savefig('mutual_information_matrix.png')
    plt.show()
    plt.close()

    # Plot 3: Eigenvalue Ratios and Curvature
    plt.figure(figsize=(10, 6))
    plt.plot(ns, r1s, label='λ₂/λ₁ (1D→2D)')
    plt.plot(ns, r2s, label='λ₃/λ₂ (2D→3D)')
    plt.plot(ns, r3s, label='λ₄/λ₃ (3D→4D)')
    plt.plot(ns, r4s, label='λ₅/λ₄ (4D→5D)')
    plt.axhline(y=1.3, color='r', linestyle='--', label='1D→2D Threshold (1.3)')
    plt.axhline(y=1.15, color='g', linestyle='--', label='2D→3D Threshold (1.15)')
    plt.axhline(y=1.01, color='b', linestyle='--', label='3D→4D Threshold (1.01)')
    plt.axhline(y=1.00, color='m', linestyle='--', label='4D→5D Threshold (1.00)')
    if results.get('2D') is not None:
        plt.axvline(x=results.get('2D'), color='r', linestyle=':', label=f'1D→2D at n={results.get("2D")}')
    if results.get('3D') is not None:
        plt.axvline(x=results.get('3D'), color='g', linestyle=':', label=f'2D→3D at n={results.get("3D")}')
    if results.get('4D') is not None:
        plt.axvline(x=results.get('4D'), color='b', linestyle=':', label=f'3D→4D at n={results.get("4D")}')
    if results.get('5D') is not None:
        plt.axvline(x=results.get('5D'), color='m', linestyle=':', label=f'4D→5D at n={results.get("5D")}')
    plt.xlabel('Number of Distinctions (n)')
    plt.ylabel('Eigenvalue Ratios')
    plt.title('Eigenvalue Ratios for Dimensional Transitions')
    plt.grid(True)
    plt.legend()
    plt.savefig('eigenvalue_ratios.png')
    plt.show()
    plt.close()

    # Plot 4: Informational Curvature
    plt.figure(figsize=(10, 6))
    plt.plot(ns, R_ns, label='R_n')
    plt.axhline(y=1.3, color='r', linestyle='--', label='1D→2D Threshold (1.3)')
    plt.axhline(y=2.2, color='g', linestyle='--', label='2D→3D Threshold (2.2)')
    plt.axhline(y=3.0, color='b', linestyle='--', label='3D→4D Threshold (3.0)')
    plt.axhline(y=4.0, color='m', linestyle='--', label='4D→5D Threshold (4.0)')
    if results.get('2D') is not None:
        plt.axvline(x=results.get('2D'), color='r', linestyle=':', label=f'1D→2D at n={results.get("2D")}')
    if results.get('3D') is not None:
        plt.axvline(x=results.get('3D'), color='g', linestyle=':', label=f'2D→3D at n={results.get("3D")}')
    if results.get('4D') is not None:
        plt.axvline(x=results.get('4D'), color='b', linestyle=':', label=f'3D→4D at n={results.get("4D")}')
    if results.get('5D') is not None:
        plt.axvline(x=results.get('5D'), color='m', linestyle=':', label=f'4D→5D at n={results.get("5D")}')
    plt.xlabel('Number of Distinctions (n)')
    plt.ylabel('Informational Curvature (R_n)')
    plt.title('Informational Curvature Evolution')
    plt.grid(True)
    plt.legend()
    plt.savefig('informational_curvature.png')
    plt.show()
    plt.close()

    # Plot 5: 2D Embedding at 1D→2D Transition
    if results.get('2D') is not None:
        G_2D = G_full[:results['2D']]
        embedding_2D = compute_embedding(G_2D, 2)
        plt.figure(figsize=(8, 6))
        plt.scatter(embedding_2D[:, 0], embedding_2D[:, 1], c='blue', label='Distinctions')
        plt.xlabel('X (Eigenvector 1)')
        plt.ylabel('Y (Eigenvector 2)')
        plt.title(f'2D Spectral Embedding at n={results["2D"]} (1D→2D)')
        plt.grid(True)
        plt.legend()
        plt.savefig('embedding_2D.png')
        plt.show()
        plt.close()

    # Plot 6: 3D Embedding at 2D→3D Transition
    if results.get('3D') is not None:
        G_3D = G_full[:results['3D']]
        embedding_3D = compute_embedding(G_3D, 3)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding_3D[:, 0], embedding_3D[:, 1], embedding_3D[:, 2], c='red', label='Distinctions')
        ax.set_xlabel('X (Eigenvector 1)')
        ax.set_ylabel('Y (Eigenvector 2)')
        ax.set_zlabel('Z (Eigenvector 3)')
        ax.set_title(f'3D Spectral Embedding at n={results["3D"]} (2D→3D)')
        plt.legend()
        plt.savefig('embedding_3D.png')
        plt.show()
        plt.close()

def print_summary(results, metrics_history):
    """Print a summary table of essential calculated values"""
    ns, d_mins, l_infos, R_ns, r1s, r2s, r3s, r4s = metrics_history
    print("\nSummary of Essential Calculated Values:")
    print("-" * 70)
    print(f"{'n':>5} | {'d_min':>8} | {'l_info':>8} | {'R_n':>8} | {'λ₂/λ₁':>8} | {'λ₃/λ₂':>8} | {'λ₄/λ₃':>8} | {'λ₅/λ₄':>8}")
    print("-" * 70)
    
    # Print at 2D transition
    if results["2D"] is not None:
        idx = results["2D"] - 1
        print(f"{ns[idx]:>5} | {d_mins[idx]:>8.3f} | {l_infos[idx]:>8.3f} | {R_ns[idx]:>8.3f} | {r1s[idx]:>8.3f} | {r2s[idx]:>8.3f} | {r3s[idx]:>8.3f} | {r4s[idx]:>8.3f} (2D Transition)")
    
    # Print at 3D transition
    if results["3D"] is not None:
        idx = results["3D"] - 1
        print(f"{ns[idx]:>5} | {d_mins[idx]:>8.3f} | {l_infos[idx]:>8.3f} | {R_ns[idx]:>8.3f} | {r1s[idx]:>8.3f} | {r2s[idx]:>8.3f} | {r3s[idx]:>8.3f} | {r4s[idx]:>8.3f} (3D Transition)")
    
    # Print at 4D transition
    if results["4D"] is not None:
        idx = results["4D"] - 1
        print(f"{ns[idx]:>5} | {d_mins[idx]:>8.3f} | {l_infos[idx]:>8.3f} | {R_ns[idx]:>8.3f} | {r1s[idx]:>8.3f} | {r2s[idx]:>8.3f} | {r3s[idx]:>8.3f} | {r4s[idx]:>8.3f} (4D Transition)")
    
    # Print at 5D transition
    if results["5D"] is not None:
        idx = results["5D"] - 1
        print(f"{ns[idx]:>5} | {d_mins[idx]:>8.3f} | {l_infos[idx]:>8.3f} | {R_ns[idx]:>8.3f} | {r1s[idx]:>8.3f} | {r2s[idx]:>8.3f} | {r3s[idx]:>8.3f} | {r4s[idx]:>8.3f} (5D Transition)")
    
    # Print at final n
    idx = len(ns) - 1
    print(f"{ns[idx]:>5} | {d_mins[idx]:>8.3f} | {l_infos[idx]:>8.3f} | {R_ns[idx]:>8.3f} | {r1s[idx]:>8.3f} | {r2s[idx]:>8.3f} | {r3s[idx]:>8.3f} | {r4s[idx]:>8.3f} (Final)")

def print_validation(G, results):
    """Print enhanced validation parameters to confirm compliance with the framework"""
    is_valid, entropy = validate_golomb(G)
    print("\nValidation Parameters:")
    print("-" * 80)
    print(f"Golomb Ruler Validity (Axiom II): {'Valid' if is_valid else 'Invalid'}")
    print(f"Entropy (S_n = n(n-1)/2, Appendix C): {entropy}")
    
    # Validate temporal order (Axiom III)
    temporal_valid = np.all(np.diff(G) > 0)
    print(f"Temporal Order (Axiom III, G[i] < G[i+1]): {'Valid' if temporal_valid else 'Invalid'}")
    
    # Validate 1D→2D transition
    if results["2D"] is not None:
        G_2D = G[:results["2D"]]
        d_min, l_info, R_n, W = compute_metrics(G_2D)
        _, _, _, _, r1, r2, r3, r4 = check_transitions(G_2D, d_min, l_info, R_n)
        
        # Compute energy functional (Axiom V)
        n = len(G_2D)
        d_ij = 1 / (1 + W)
        np.fill_diagonal(d_ij, np.inf)
        E_n = np.sum(np.abs(1 / d_ij[d_ij < np.inf]**2 - 1 / l_info**2)) / 2
        
        # Compute spectral gaps
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, 5])
        λ0, λ1, λ2, λ3, λ4, λ5 = eigenvalues
        gap1 = λ1
        gap2 = λ2 - λ1
        gap3 = λ3 - λ2
        gap4 = λ4 - λ3
        gap5 = λ5 - λ4
        
        # Compute embedding distortion in 2D
        embedding_2D = compute_embedding(G_2D, 2)
        euclidean_dists = np.sqrt(np.sum((embedding_2D[:, None] - embedding_2D)**2, axis=2))
        np.fill_diagonal(d_ij, 0)
        distortion = np.mean((euclidean_dists - d_ij)**2 / (d_ij**2 + 1e-16))
        
        print(f"\n1D→2D Transition at n={results['2D']}:")
        print(f"  λ₂/λ₁ = {r1:.3f} (> 1.3: {'Valid' if r1 > 1.3 else 'Invalid'})")
        print(f"  R_n = {R_n:.3f} (> 1.3: {'Valid' if R_n > 1.3 else 'Invalid'})")
        print(f"  Energy Functional (Axiom V): E_n = {E_n:.3f} (>= 0: {'Valid' if E_n >= 0 else 'Invalid'})")
        print(f"  Spectral Gaps (Annex H): λ₁ = {gap1:.3f}, λ₂-λ₁ = {gap2:.3f}, λ₃-λ₂ = {gap3:.3f}, λ₄-λ₃ = {gap4:.3f}, λ₅-λ₄ = {gap5:.3f}")
        print(f"  2D Embedding Distortion (Annex D): {distortion:.3f}")
    
    # Validate 2D→3D transition
    if results["3D"] is not None:
        G_3D = G[:results["3D"]]
        d_min, l_info, R_n, W = compute_metrics(G_3D)
        _, _, _, _, r1, r2, r3, r4 = check_transitions(G_3D, d_min, l_info, R_n)
        
        # Compute energy functional
        n = len(G_3D)
        d_ij = 1 / (1 + W)
        np.fill_diagonal(d_ij, np.inf)
        E_n = np.sum(np.abs(1 / d_ij[d_ij < np.inf]**2 - 1 / l_info**2)) / 2
        
        # Compute spectral gaps
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, 5])
        λ0, λ1, λ2, λ3, λ4, λ5 = eigenvalues
        gap1 = λ1
        gap2 = λ2 - λ1
        gap3 = λ3 - λ2
        gap4 = λ4 - λ3
        gap5 = λ5 - λ4
        
        # Compute embedding distortion in 3D
        embedding_3D = compute_embedding(G_3D, 3)
        euclidean_dists = np.sqrt(np.sum((embedding_3D[:, None] - embedding_3D)**2, axis=2))
        np.fill_diagonal(d_ij, 0)
        distortion = np.mean((euclidean_dists - d_ij)**2 / (d_ij**2 + 1e-16))
        
        print(f"\n2D→3D Transition at n={results['3D']}:")
        print(f"  λ₃/λ₂ = {r2:.3f} (> 1.15: {'Valid' if r2 > 1.15 else 'Invalid'})")
        print(f"  R_n = {R_n:.3f} (> 2.2: {'Valid' if R_n > 2.2 else 'Invalid'})")
        print(f"  Energy Functional (Axiom V): E_n = {E_n:.3f} (>= 0: {'Valid' if E_n >= 0 else 'Invalid'})")
        print(f"  Spectral Gaps (Annex H): λ₁ = {gap1:.3f}, λ₂-λ₁ = {gap2:.3f}, λ₃-λ₂ = {gap3:.3f}, λ₄-λ₃ = {gap4:.3f}, λ₅-λ₄ = {gap5:.3f}")
        print(f"  3D Embedding Distortion (Annex D): {distortion:.3f}")
    
    # Validate 3D→4D transition
    if results["4D"] is not None:
        G_4D = G[:results["4D"]]
        d_min, l_info, R_n, W = compute_metrics(G_4D)
        _, _, _, _, r1, r2, r3, r4 = check_transitions(G_4D, d_min, l_info, R_n)
        
        # Compute energy functional
        n = len(G_4D)
        d_ij = 1 / (1 + W)
        np.fill_diagonal(d_ij, np.inf)
        E_n = np.sum(np.abs(1 / d_ij[d_ij < np.inf]**2 - 1 / l_info**2)) / 2
        
        # Compute spectral gaps
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, 5])
        λ0, λ1, λ2, λ3, λ4, λ5 = eigenvalues
        gap1 = λ1
        gap2 = λ2 - λ1
        gap3 = λ3 - λ2
        gap4 = λ4 - λ3
        gap5 = λ5 - λ4
        
        # Compute embedding distortion in 4D
        embedding_4D = compute_embedding(G_4D, 4)
        euclidean_dists = np.sqrt(np.sum((embedding_4D[:, None] - embedding_4D)**2, axis=2))
        np.fill_diagonal(d_ij, 0)
        distortion = np.mean((euclidean_dists - d_ij)**2 / (d_ij**2 + 1e-16))
        
        print(f"\n3D→4D Transition at n={results['4D']}:")
        print(f"  λ₄/λ₃ = {r3:.3f} (> 1.01: {'Valid' if r3 > 1.01 else 'Invalid'})")
        print(f"  R_n = {R_n:.3f} (> 3.0: {'Valid' if R_n > 3.0 else 'Invalid'})")
        print(f"  Energy Functional (Axiom V): E_n = {E_n:.3f} (>= 0: {'Valid' if E_n >= 0 else 'Invalid'})")
        print(f"  Spectral Gaps (Annex H): λ₁ = {gap1:.3f}, λ₂-λ₁ = {gap2:.3f}, λ₃-λ₂ = {gap3:.3f}, λ₄-λ₃ = {gap4:.3f}, λ₅-λ₄ = {gap5:.3f}")
        print(f"  4D Embedding Distortion (Annex D): {distortion:.3f}")
    
    # Validate 4D→5D transition
    if results["5D"] is not None:
        G_5D = G[:results["5D"]]
        d_min, l_info, R_n, W = compute_metrics(G_5D)
        _, _, _, _, r1, r2, r3, r4 = check_transitions(G_5D, d_min, l_info, R_n)
        
        # Compute energy functional
        n = len(G_5D)
        d_ij = 1 / (1 + W)
        np.fill_diagonal(d_ij, np.inf)
        E_n = np.sum(np.abs(1 / d_ij[d_ij < np.inf]**2 - 1 / l_info**2)) / 2
																	 
        # Compute spectral gaps
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, 5])
        λ0, λ1, λ2, λ3, λ4, λ5 = eigenvalues
        gap1 = λ1
        gap2 = λ2 - λ1
        gap3 = λ3 - λ2
        gap4 = λ4 - λ3
        gap5 = λ5 - λ4
        
        # Compute embedding distortion in 5D
        embedding_5D = compute_embedding(G_5D, 5)
        euclidean_dists = np.sqrt(np.sum((embedding_5D[:, None] - embedding_5D)**2, axis=2))
        np.fill_diagonal(d_ij, 0)
        distortion = np.mean((euclidean_dists - d_ij)**2 / (d_ij**2 + 1e-16))
        
        print(f"\n4D→5D Transition at n={results['5D']}:")
        print(f"  λ₅/λ₄ = {r4:.3f} (> 1.00: {'Valid' if r4 > 1.00 else 'Invalid'})")
        print(f"  R_n = {R_n:.3f} (> 4.0: {'Valid' if R_n > 4.0 else 'Invalid'})")
        print(f"  Energy Functional (Axiom V): E_n = {E_n:.3f} (>= 0: {'Valid' if E_n >= 0 else 'Invalid'})")
        print(f"  Spectral Gaps (Annex H): λ₁ = {gap1:.3f}, λ₂-λ₁ = {gap2:.3f}, λ₃-λ₂ = {gap3:.3f}, λ₄-λ₃ = {gap4:.3f}, λ₅-λ₄ = {gap5:.3f}")
        print(f"  5D Embedding Distortion (Annex D): {distortion:.3f}")

def simulate(n_max):
    """Efficient simulation with reused Golomb ruler prefix, enhanced with plotting and validation"""
    results = {"2D": None, "3D": None, "4D": None, "5D": None}
    G_full = generate_golomb(n_max)
    
    # Store metrics for plotting
    ns = []
    d_mins = []
    l_infos = []
    R_ns = []
    r1s = []
    r2s = []
    r3s = []
    r4s = []

    for n in range(1, n_max + 1):
        G = G_full[:n]
        d_min, l_info, R_n, _ = compute_metrics(G)
        t2d, t3d, t4d, t5d, r1, r2, r3, r4 = check_transitions(G, d_min, l_info, R_n)

        ns.append(n)
        d_mins.append(d_min)
        l_infos.append(l_info)
        R_ns.append(R_n)
        r1s.append(r1)
        r2s.append(r2)
        r3s.append(r3)
        r4s.append(r4)

        if t2d and results["2D"] is None:
            results["2D"] = n                       
        if t3d and results["2D"] is not None and results["3D"] is None:
            results["3D"] = n
        if t4d and results["3D"] is not None and results["4D"] is None:
            results["4D"] = n
        if t5d and results["4D"] is not None and results["5D"] is None:
            results["5D"] = n

        if n % 10 == 0:
            print(f"Progress: n={n}, d_min={d_min:.3f}, l_info={l_info:.3f}, R_n={R_n:.3f}")
    
    # Generate and display plots
    plot_results(G_full, results, (ns, d_mins, l_infos, R_ns, r1s, r2s, r3s, r4s))
    
    # Print summary
    print_summary(results, (ns, d_mins, l_infos, R_ns, r1s, r2s, r3s, r4s))
    
    # Print validation
    print_validation(G_full, results)
    
    return results

# Run simulation                                                                  
print("Starting simulation...")
results = simulate(1000)
print("\nFinal Results:")
print(f"1D→2D transition at n={results['2D']}")
print(f"2D→3D transition at n={results['3D']}")
print(f"3D→4D transition at n={results['4D']}")
print(f"4D→5D transition at n={results['5D']}")
