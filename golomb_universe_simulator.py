import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.cluster import SpectralClustering
from scipy.stats import entropy
from itertools import combinations

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
    return G

# Simulate distinction data (binary strings per Golomb point)
def simulate_distinction_data(G, length=500):
    np.random.seed(42)
    return {x: np.random.randint(0, 2, length) for x in G}

# Compute mutual information between two binary vectors
def mutual_information(x, y, bins=2):
    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    joint_prob = joint_hist / np.sum(joint_hist)
    x_marg = np.sum(joint_prob, axis=1)
    y_marg = np.sum(joint_prob, axis=0)
    nz_joint = joint_prob > 0
    return np.sum(joint_prob[nz_joint] * np.log(joint_prob[nz_joint] /
                 (x_marg[:, None] * y_marg[None, :])[nz_joint]))

# Build mutual information matrix
def compute_mi_matrix(data):
    keys = list(data.keys())                                     
    n = len(keys)
    mi_matrix = np.zeros((n, n))
    for i in range(n):     
        for j in range(i + 1, n):
            mi = mutual_information(data[keys[i]], data[keys[j]])
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    return mi_matrix, keys

# Plot MI matrix as heatmap
def plot_mi_heatmap(mi_matrix, keys, title="Mutual Information Matrix"):                 
    plt.figure(figsize=(10, 8))
    sns.heatmap(mi_matrix, xticklabels=False, yticklabels=False, cmap="magma")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Spectral clustering on MI graph (optional)
def spectral_cluster(mi_matrix, n_clusters=3):
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    return sc.fit_predict(mi_matrix)

# Optional graph visualization
def plot_mi_graph(mi_matrix, labels, keys):
    G = nx.Graph()
    for i, k in enumerate(keys):
        G.add_node(k, cluster=labels[i])
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if mi_matrix[i, j] > 0:
                G.add_edge(keys[i], keys[j], weight=mi_matrix[i, j])

    pos = nx.spring_layout(G, seed=42)
    colors = [labels[i] for i in range(len(keys))]
    nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.Set2,
            node_size=500, edge_color='gray')
    plt.title("Informational Structure Graph")
    plt.show()

# Main entry point                           
def main():                                  
    n = 300  # number of distinctions
    G = generate_golomb_sequence(n)
    data = simulate_distinction_data(G, length=500)
    mi_matrix, keys = compute_mi_matrix(data)                         

    plot_mi_heatmap(mi_matrix, keys)

    # Optional: cluster and visualize
    #labels = spectral_cluster(mi_matrix, n_clusters=3)
    #plot_mi_graph(mi_matrix, labels, keys)
												
    # Histogram of MI values
    upper_triangle_mi_values = mi_matrix[np.triu_indices(mi_matrix.shape[0], k=1)]
    plt.figure(figsize=(10, 8))
    sns.histplot(upper_triangle_mi_values, kde=True, bins=20)
    plt.title("Histogram of Mutual Information Values")
    plt.xlabel("Mutual Information ($MI$)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
