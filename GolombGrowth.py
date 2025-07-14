import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

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
        # Check against existing distinctions (D) and for internal duplicates among new_diffs
        # The loop below combines the check for `diff in D` and internal uniqueness for `new_diffs`
        temp_new_diffs_set = set()
        for diff in new_diffs:
            if diff in D or diff in temp_new_diffs_set: # Embodies Axiom II: Irreducible Uniqueness
                is_unique = False
                break
            temp_new_diffs_set.add(diff)

        while not is_unique: # This loop structure iterates 'm' until a unique candidate is found
            m += 1
            new_diffs = [abs(m - x) for x in G]
            is_unique = True # Reset for the new 'm'
            temp_new_diffs_set = set()
            for diff in new_diffs:
                if diff in D or diff in temp_new_diffs_set:
                    is_unique = False
                    break
                temp_new_diffs_set.add(diff)
            
        # If 'is_unique' is True after the loops, 'm' is valid
        G.append(m) # This is the "morphism" f: G_k -> G_{k+1} (Axiom III)
        D.update(temp_new_diffs_set) # Use the set of valid new diffs

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
    This is defined as the sum of inverse distances between all unique pairs.
    A lower energy (higher sum of inverse distances) implies a more 'compact'
    or 'efficient' ruler with smaller distances contributing more significantly.
    """
    if not G or len(G) < 2:
        return float('inf') # Or 0.0 depending on desired interpretation for trivial rulers
    return sum(1.0 / abs(a - b) for a, b in combinations(G, 2))

def plot_golomb_graph(G: list[int]):
    """
    Create and visualize the Golomb ruler as a graph where nodes are marks
    and edges represent the unique distances between them.
    Uses NetworkX and Matplotlib.
    """
    # Create graph
    g = nx.Graph()
    g.add_nodes_from(G)
    # Add edges with their weights (distances)
    edges = [(a, b, {'weight': abs(a-b)}) for a, b in combinations(G, 2)]
    g.add_edges_from(edges)
    
    # Calculate positions (circular layout provides good separation for this type of graph)
    pos = nx.circular_layout(g)
    
    # Draw the graph
    plt.figure(figsize=(10, 10)) # Increased figure size for better visibility
    
    # Draw nodes
    nx.draw_networkx_nodes(g, pos, node_size=700,
                           node_color='lightblue', alpha=0.9, linewidths=1.0, edgecolors='gray')
    
    # Draw edges
    nx.draw_networkx_edges(g, pos, width=1.5, alpha=0.7, edge_color='darkgray')
    
    # Draw node labels
    nx.draw_networkx_labels(g, pos, font_size=14, 
                             font_weight='bold', font_color='black')
    
    # Draw edge labels (distances)
    edge_labels = {(u, v): d['weight'] for u, v, d in g.edges(data=True)}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                  font_color='darkgreen', font_size=10,
                                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.title(f"Golomb Graph for Sequence {G}\nEnergy: {energy_fn(G):.2f}, Entropy: {entropy_fn(len(G))}", fontsize=12)
    plt.axis('off') # Hide axes
    plt.gca().set_aspect('equal')  # Set equal aspect rati
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()

# --- Example Usage (Main Execution Block) ---
if __name__ == "__main__":
    n_marks = 10 # Example for an n-mark Golomb ruler
    golomb_sequence = golomb_grow(n_marks)
    print(f"Golomb sequence for {n_marks} marks: {golomb_sequence}")
    
    entropy_val = entropy_fn(len(golomb_sequence))
    print(f"Combinatorial Entropy (max distinctions): {entropy_val}")
    
    energy_val = energy_fn(golomb_sequence)
    print(f"Energy (sum of inverse distances): {energy_val:.4f}")
    
    # Visualize the generated Golomb ruler
    plot_golomb_graph(golomb_sequence)

    
