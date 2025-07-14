import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from math import log

def golomb_grow(n: int) -> list[int]:
    """
    Generates a Golomb ruler of 'n' marks using an iterative growth approach,
    as derived from the axiomatic framework.
    """
    G = [0]  # The Golomb ruler sequence, initialized with 0 (Axiom 0)
    D = set()  # Set to store all unique distinctions found so far (Axiom II)

    while len(G) < n:
        m = G[-1] + 1  # Start searching for the next mark from the previous mark + 1
        new_diffs = [abs(m - x) for x in G]
        is_unique = True
        temp_new_diffs_set = set()
        for diff in new_diffs:
            if diff in D or diff in temp_new_diffs_set:  # Embodies Axiom II: Irreducible Uniqueness
                is_unique = False
                break
            temp_new_diffs_set.add(diff)

        while not is_unique:  # Iterate 'm' until a unique candidate is found
            m += 1
            new_diffs = [abs(m - x) for x in G]
            is_unique = True
            temp_new_diffs_set = set()
            for diff in new_diffs:
                if diff in D or diff in temp_new_diffs_set:
                    is_unique = False
                    break
                temp_new_diffs_set.add(diff)
            
        G.append(m)  # Morphism f: G_k -> G_{k+1} (Axiom III)
        D.update(temp_new_diffs_set)  # Update set of unique differences

    return G

def entropy_fn(n: int) -> int:
    """
    Calculates the maximum possible number of unique distinctions for 'n' marks,
    representing combinatorial entropy for the system.
    """
    return n * (n - 1) // 2

def energy_fn(G: list[int]) -> float:
    """
    Calculates an 'energy' metric for a given Golomb ruler, aligned with Axiom V.
    Sum of inverse distances between all unique pairs.
    """
    if not G or len(G) < 2:
        return float('inf')  # Or 0.0 depending on interpretation
    return sum(1.0 / abs(a - b) for a, b in combinations(G, 2))

def estimated_max_energy(n: int) -> float:
    """
    Estimate the maximum theoretical energy for n marks,
    assuming distances are the first n(n-1)/2 integers.
    Uses harmonic number approximation: H_m ≈ ln(m) + γ,
    where m = number of unique pairs = n(n-1)/2.
    """
    gamma = 0.5772156649  # Euler–Mascheroni constant
    m = n * (n - 1) // 2  # number of unique pairs
    return log(m) + gamma

def plot_golomb_graph(G: list[int]):
    """
    Visualize the Golomb ruler as a graph with nodes as marks and edges as distances.
    Displays actual energy and estimated maximum energy in the title.
    """
    g = nx.Graph()
    g.add_nodes_from(G)
    edges = [(a, b, {'weight': abs(a-b)}) for a, b in combinations(G, 2)]
    g.add_edges_from(edges)
    
    pos = nx.circular_layout(g)
    
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(g, pos, node_size=700,
                           node_color='lightblue', alpha=0.9, linewidths=1.0, edgecolors='gray')
    nx.draw_networkx_edges(g, pos, width=1.5, alpha=0.7, edge_color='darkgray')
    nx.draw_networkx_labels(g, pos, font_size=14, font_weight='bold', font_color='black')
    
    edge_labels = {(u, v): d['weight'] for u, v, d in g.edges(data=True)}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                 font_color='darkgreen', font_size=10,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3'))
    
    actual_energy = energy_fn(G)
    max_energy = estimated_max_energy(len(G))
    plt.title(f"Golomb Graph for Sequence {G}\n'Energy': {actual_energy:.4f}, Estimated Max 'Energy': {max_energy:.4f}", fontsize=12)
    
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

# --- Example Usage (Main Execution Block) ---
if __name__ == "__main__":
    n_marks = 10  # Example for an n-mark Golomb ruler
    golomb_sequence = golomb_grow(n_marks)
    print(f"Golomb sequence for {n_marks} marks: {golomb_sequence}")
    
    entropy_val = entropy_fn(len(golomb_sequence))
    print(f"Combinatorial Entropy (max distinctions): {entropy_val}")
    
    energy_val = energy_fn(golomb_sequence)
    print(f"Energy (sum of inverse distances): {energy_val:.4f}")
    
    max_energy_val = estimated_max_energy(len(golomb_sequence))
    print(f"Estimated max energy for {len(golomb_sequence)} marks: {max_energy_val:.4f}")
    
    plot_golomb_graph(golomb_sequence)
