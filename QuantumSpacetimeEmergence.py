import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from skimage.measure import block_reduce

class QuantumGolombSpacetime:
    def __init__(self, initial_marks=None):
        self.marks = sorted(initial_marks) if initial_marks else [0, 1]
        self.distances = self._compute_distances(self.marks)
        self.history = [self.marks.copy()]
        self.causal_net = None
        self.matter_density = [0.0] * len(self.marks)
        self.curvatures = []

    def _compute_distances(self, marks):
        return {abs(a - b) for i, a in enumerate(marks) for b in marks[i+1:]}

    def _is_valid_addition(self, candidate):
        new_dists = {abs(candidate - mark) for mark in self.marks}
        return new_dists.isdisjoint(self.distances)

# -----------------------------------------------------------------------------
# DOCTRINE NOTE: Emergent Golomb Spacetime
# -----------------------------------------------------------------------------
# This simulation does not aim to construct optimal or minimal Golomb rulers.
# Instead, it evolves a set of integers (marks) under the constraint that all
# pairwise distances remain unique — the Golomb condition.
#
# These marks are interpreted as temporal events in a growing spacetime, influenced
# by physical analogs such as matter density, curvature, and quantum fluctuations.
#
# The goal is not combinatorial perfection, but physical emergence. Golomb logic
# serves as a structural law — not as an optimization target.
#
# In short: this code grows Golomb spacetime, not Golomb rulers.
# -----------------------------------------------------------------------------
    
    def quantum_growth(self, max_marks=30, temperature=0.02, search_limit=1000):
        while len(self.marks) < max_marks:
            current_max = self.marks[-1]
            
            valid_candidates = []
            candidate = current_max + 1
            while candidate <= current_max + search_limit:
                if self._is_valid_addition(candidate):
                    valid_candidates.append(candidate)
                candidate += 1
                
            if not valid_candidates:
                print(f"Stopped growing at {len(self.marks)} marks")
                break
                
            positions = np.arange(len(valid_candidates))
            probabilities = np.exp(-positions/temperature)
            probabilities /= np.sum(probabilities)
            
            chosen = np.random.choice(valid_candidates, p=probabilities)
            
            if len(self.marks) >= 3:
                chosen = self.matter_curvature_coupling(chosen)
            
            new_dists = {abs(chosen - mark) for mark in self.marks}
            self.distances.update(new_dists)
            self.marks.append(chosen)
            self.history.append(self.marks.copy())
            self.matter_density.append(0.0)
            self.update_matter_curvature()
            
        return self.marks

    def matter_curvature_coupling(self, candidate):
        potential = 0.0
        for i, mark in enumerate(self.marks):
            dist = candidate - mark
            if dist > 0:
                potential += self.matter_density[i] / (dist**2 + 1e-8)
        
        shift = 0.5 * potential * np.random.normal()
        shifted_candidate = candidate + shift
        new_candidate = int(np.round(shifted_candidate))
        
        if self._is_valid_addition(new_candidate):
            return new_candidate
        
        for offset in [-1, 1, -2, 2, -3, 3]:
            trial = new_candidate + offset
            if self._is_valid_addition(trial):
                return trial
                
        return candidate

    def update_matter_curvature(self):
        if len(self.marks) < 3:
            return
            
        x, y, r, theta = self.embed_polar()
        curvatures = self.estimate_curvature(x, y)
        self.curvatures = curvatures
        
        matter_arr = np.array(self.matter_density)
        
        if len(curvatures) == len(matter_arr):
            curv_arr = np.array(curvatures)
            matter_arr = 0.1 * np.abs(curv_arr) + 0.01 * np.random.randn(len(curv_arr))
            matter_arr = np.maximum(matter_arr, 0)
        
        self.matter_density = matter_arr.tolist()

    def embed_polar(self, log_scaling=True):
        n = len(self.marks)
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = np.array(self.marks)
        
        if log_scaling:
            r = np.log1p(r)
            
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, r, theta

    def compute_mass_density(self, x, y, size=512):
        max_extent = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.1
        grid_range = [[-max_extent, max_extent], [-max_extent, max_extent]]
        
        H, xedges, yedges = np.histogram2d(x, y, bins=size, range=grid_range)
        H_smoothed = gaussian_filter(H, sigma=2.0)
        
        return H_smoothed, (xedges, yedges)

    def estimate_box_dimension(self, Z, epsilons=None):
        if epsilons is None:
            max_dim = min(Z.shape)
            epsilons = [2**i for i in range(1, int(np.log2(max_dim)))]
        
        sizes = []
        valid_eps = []
        
        for eps in epsilons:
            if Z.shape[0] >= eps and Z.shape[1] >= eps:
                crop_size = (Z.shape[0] // eps) * eps
                cropped = Z[:crop_size, :crop_size]
                reduced = block_reduce(cropped, block_size=(eps, eps), func=np.mean)
                count = np.sum(reduced > 0.1 * np.max(Z))
                sizes.append(count)
                valid_eps.append(eps)
        
        if len(sizes) < 3:
            print("Not enough valid epsilon values for dimension estimation")
            return 0.0
        
        logs = np.log(np.array(sizes))
        logs_eps = np.log(1.0 / np.array(valid_eps))
        slope, intercept = np.polyfit(logs_eps, logs, 1)
        
        fig = plt.figure(figsize=(8, 5))
        fig.canvas.manager.set_window_title("Fractal Dimension Estimation")

        plt.plot(logs_eps, logs, 'bo-')
        plt.plot(logs_eps, slope*logs_eps + intercept, 'r--', 
                 label=f'Slope (dimension) = {slope:.3f}')
        plt.xlabel('log(1/ε)')
        plt.ylabel('log(N(ε))')
        plt.title('Fractal Dimension Estimation')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return slope

    def build_causal_network(self, theta):
        G = nx.DiGraph()
        n = len(self.marks)
        
        for i, (r, t) in enumerate(zip(self.marks, theta)):
            G.add_node(i, radius=r, theta=t, matter=self.matter_density[i])
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.marks[j] > self.marks[i]:
                    dt = self.marks[j] - self.marks[i]
                    dtheta = min(abs(theta[j] - theta[i]), 
                                2*np.pi - abs(theta[j] - theta[i]))
                    weight = self.matter_density[i] / (dt * (dtheta + 0.1))
                    G.add_edge(i, j, weight=weight, dt=dt, dtheta=dtheta)
        
        self.causal_net = G
        return G

    def analyze_causal_net(self):
        if self.causal_net is None:
            print("Build causal network first!")
            return
        
        n_nodes = self.causal_net.number_of_nodes()
        n_edges = self.causal_net.number_of_edges()
        density = n_edges / (n_nodes*(n_nodes-1))
        
        causally_connected = 0
        path_lengths = []
        for i in self.causal_net.nodes:
            for j in self.causal_net.nodes:
                if i != j and nx.has_path(self.causal_net, i, j):
                    causally_connected += 1
                    path_lengths.append(nx.shortest_path_length(self.causal_net, i, j, weight='dt'))
        
        avg_path = np.mean(path_lengths) if path_lengths else 0
        
        print("\n" + "="*50)
        print("Causal Network Analysis")
        print("="*50)
        print(f"Nodes: {n_nodes}, Edges: {n_edges}")
        print(f"Connection density: {density:.4f}")
        print(f"Causally connected pairs: {causally_connected}/{n_nodes*(n_nodes-1)}")
        print(f"Average causal time distance: {avg_path:.2f}")
        
        degrees = [d for n, d in self.causal_net.out_degree()]
        
        fig = plt.figure(figsize=(12, 5))
        fig.canvas.manager.set_window_title("Causal Network Metrics")

        plt.subplot(121)
        plt.bar(range(n_nodes), degrees, color='skyblue')
        plt.title("Out-Degree vs Temporal Position")
        plt.xlabel("Event Index")
        plt.ylabel("Out-Degree")
        plt.grid(True)
        
        plt.subplot(122)
        plt.scatter(self.matter_density, degrees, c=self.marks, cmap='viridis')
        plt.colorbar(label='Temporal Position')
        plt.title("Matter Density vs Causal Influence")
        plt.xlabel("Matter Density")
        plt.ylabel("Out-Degree")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\nCausal Structure Insights:")
        print(f"- First event connects to {degrees[0]} future events")
        print(f"- Last event has {degrees[-1]} outgoing connections")
        print(f"- Middle event (index {n_nodes//2}) connects to {degrees[n_nodes//2]} future events")
        
        if len(self.matter_density) == len(degrees):
            correlation = np.corrcoef(self.matter_density, degrees)[0,1]
            print(f"Matter-Degree Correlation: {correlation:.3f}")
        else:
            correlation = 0.0
            
        self.avg_causal_path = avg_path
        self.matter_degree_corr = correlation

    def estimate_curvature(self, x, y):
        coords = np.column_stack([x, y])
        if len(coords) < 3:
            return np.array([])
            
        nbrs = NearestNeighbors(n_neighbors=3).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        curvatures = []
        for i in range(len(coords)):
            n1, n2 = indices[i][1], indices[i][2]
            p0, p1, p2 = coords[i], coords[n1], coords[n2]
            
            v1 = p1 - p0
            v2 = p2 - p0
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            curvature = (np.pi - angle) / (distances[i][1] * distances[i][2])
            curvatures.append(curvature)
            
        return np.array(curvatures)

    def print_physics_insights(self):
        print("\n" + "="*60)
        print("QUANTUM SPACETIME PHYSICS INSIGHTS")
        print("="*60)
        
        if hasattr(self, 'fractal_dim'):
            if 2.0 < self.fractal_dim < 2.5:
                dim_insight = "SPATIAL DIMENSION ≈ 2+ε (COMPACTIFIED EXTRA DIMENSIONS)"
            elif 2.5 <= self.fractal_dim < 3.0:
                dim_insight = "EMERGENT 3D STRUCTURE (COSMIC WEB-LIKE)"
            else:
                dim_insight = "EXOTIC DIMENSIONALITY (QUANTUM FOAM)"
            print(f"Fractal Dimension {self.fractal_dim:.3f}: {dim_insight}")
        else:
            print("Fractal Dimension: Not computed")
        
        if len(self.curvatures) > 0:
            curvature_range = np.ptp(self.curvatures)
            avg_matter = np.mean(self.matter_density)
            print(f"\nMATTER-CURVATURE COUPLING:")
            print(f"- Curvature Range: {curvature_range:.4f} (High = Strong Gravity Wells)")
            print(f"- Avg Matter Density: {avg_matter:.4f}")
        else:
            print("\nMATTER-CURVATURE COUPLING: Not enough data")
        
        if len(self.marks) > 1:
            positions = np.array(self.marks)
            deviations = positions[1:] - positions[:-1] - 1
            quantum_fluct = np.mean(np.abs(deviations))
            avg_position = np.mean(positions)
            print(f"\nQUANTUM FLUCTUATIONS:")
            print(f"- Avg Position Deviation: {quantum_fluct:.4f} (Classical=0)")
            if avg_position > 0:
                print(f"- Fluctuation/Structure Ratio: {quantum_fluct/avg_position:.6f}")
        else:
            print("\nQUANTUM FLUCTUATIONS: Not enough events")
        
        if hasattr(self, 'avg_causal_path') and hasattr(self, 'matter_degree_corr'):
            causal_efficiency = self.avg_causal_path / np.ptp(self.marks) if np.ptp(self.marks) > 0 else 0
            print(f"\nCAUSAL STRUCTURE:")
            print(f"- Causal Efficiency: {causal_efficiency:.4f} (Lower = Faster Information Flow)")
            print(f"- Max Causal Horizon: {self.marks[-1] - self.marks[0]} temporal units")
            print(f"- Matter-Degree Correlation: {self.matter_degree_corr:.3f} (Expected: 0.3-0.7)")
        else:
            print("\nCAUSAL STRUCTURE: Not analyzed")
        
        if len(self.curvatures) > 0:
            total_matter = np.sum(self.matter_density)
            avg_curvature = np.mean(np.abs(self.curvatures))
            energy_balance = total_matter / (avg_curvature + 1e-8)
            print(f"\nENERGY CONSERVATION:")
            print(f"- Total Matter: {total_matter:.4f}")
            print(f"- Avg Curvature: {avg_curvature:.4f}")
            print(f"- Matter/Curvature Ratio: {energy_balance:.4f} (Expect ~constant)")
        else:
            print("\nENERGY CONSERVATION: Not enough data")

    def plot_spacetime(self, size=256):
        # Adjusted figure size for better readability and control visibility
        fig = plt.figure(figsize=(18, 9), constrained_layout=True)
        
        # Create a GridSpec with better spacing and centering
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1],
                           hspace=0.1, wspace=0.1,
                           left=0.12, right=0.88, top=0.88, bottom=0.12)
        
        # Create axes with consistent size
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Original polar embedding
        x, y, r, theta = self.embed_polar()
        sc1 = ax1.scatter(x, y, c=r, cmap='viridis', s=30, edgecolors='k', linewidth=0.5)
        ax1.set_title(f"Polar Embedding (n={len(self.marks)})", pad=15)
        ax1.set_aspect('equal')
        ax1.set_box_aspect(1)
        ax1.grid(True, alpha=0.3)
        
        # Mass density
        density, (xedges, yedges) = self.compute_mass_density(x, y, size=size)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax2.imshow(density.T, origin='lower', extent=extent, cmap='magma', 
                       aspect='auto', interpolation='bicubic')
        ax2.set_title("Mass-Energy Density Distribution", pad=15)
        ax2.set_aspect('equal')
        ax2.set_box_aspect(1)
        ax2.grid(True, alpha=0.3)
        
        # FFT analysis
        fft_mag = np.abs(fft2(density))
        fft_shift = fftshift(fft_mag)
        im_fft = ax3.imshow(np.log1p(fft_shift), cmap='inferno', 
                           aspect='auto', interpolation='bicubic')
        ax3.set_title("Fourier Space Patterns", pad=15)
        ax3.set_aspect('equal')
        ax3.set_box_aspect(1)
        ax3.grid(True, alpha=0.3)
        
        # Curvature visualization
        if len(self.curvatures) > 0:
            sc4 = ax4.scatter(x, y, c=self.curvatures, cmap='coolwarm', 
                           s=30, norm=Normalize(-1, 1), edgecolors='k', linewidth=0.5)
            ax4.set_title("Local Spacetime Curvature", pad=15)
        ax4.set_aspect('equal')
        ax4.set_box_aspect(1)
        ax4.grid(True, alpha=0.3)
        
        # Causal network visualization
        if self.causal_net is None:
            self.build_causal_network(theta)
        nx.draw_spring(self.causal_net, ax=ax5, node_size=50, 
                      with_labels=False, arrows=False, node_color='skyblue')
        ax5.set_title("Causal Network Structure", pad=15)
        ax5.set_box_aspect(1)
        ax5.grid(True, alpha=0.3)
        
        # Matter distribution visualization
        if len(self.matter_density) == len(x):
            sc6 = ax6.scatter(x, y, c=self.matter_density, cmap='plasma', 
                           s=30, edgecolors='k', linewidth=0.5)
            ax6.set_title("Matter Density Distribution", pad=15)
        ax6.set_aspect('equal')
        ax6.set_box_aspect(1)
        ax6.grid(True, alpha=0.3)
        
        # Unified colorbar system for all plots
        plots_with_colorbars = [
            (ax1, sc1, "Temporal Position"),
            (ax2, im, "Density"),
            (ax3, im_fft, "Log Magnitude"),
        ]
																						 
        # Add conditional visualizations
        if len(self.curvatures) > 0:
            plots_with_colorbars.append((ax4, sc4, "Curvature"))
        if len(self.matter_density) == len(x):
            plots_with_colorbars.append((ax6, sc6, "Matter Density"))
        
        # Create consistent colorbars
        for ax, im, label in plots_with_colorbars:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(label, fontsize=9)
            cax.tick_params(labelsize=8)
        
        # Set window title for better visibility
        fig.canvas.manager.set_window_title('Quantum Golomb Spacetime Analysis')
        
        plt.show()
        
        print("Estimating fractal dimension...")
        epsilons = [2, 4, 8, 16, 32]
        dimension = self.estimate_box_dimension(density, epsilons)
        print(f"Estimated fractal dimension: {dimension:.3f}")
        self.fractal_dim = dimension
        
        self.analyze_causal_net()
        self.print_physics_insights()

# Create and analyze quantum spacetime
simulator = QuantumGolombSpacetime(initial_marks=[0, 1])
simulator.quantum_growth(max_marks=40, temperature=0.1)
simulator.plot_spacetime(size=256)

