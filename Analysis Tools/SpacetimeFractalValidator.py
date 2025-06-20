import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
import matplotlib.gridspec as gridspec
from tqdm import tqdm

class QuantumGolombValidator:
    def __init__(self, initial_marks=None):
        self.marks = sorted(initial_marks) if initial_marks else [0, 1]
        self.distances = self._compute_distances(self.marks)
        self.history = [self.marks.copy()]
    
    def _compute_distances(self, marks):
        return {abs(a - b) for i, a in enumerate(marks) for b in marks[i+1:]}
    
    def _is_valid_addition(self, candidate):
        new_dists = {abs(candidate - mark) for mark in self.marks}
        return new_dists.isdisjoint(self.distances)
    
    def quantum_growth(self, max_marks=30, temperature=0.02, search_limit=1000):
        while len(self.marks) < max_marks:
            current_max = self.marks[-1]
            valid_candidates = [
                c for c in range(current_max + 1, current_max + search_limit + 1)
                if self._is_valid_addition(c)
            ]
            if not valid_candidates:
                break
            positions = np.arange(len(valid_candidates))
            probabilities = np.exp(-positions/temperature)
            probabilities /= probabilities.sum()
            chosen = np.random.choice(valid_candidates, p=probabilities)
            self.distances.update({abs(chosen - m) for m in self.marks})
            self.marks.append(chosen)
            self.history.append(self.marks.copy())
        return self.marks

    def embed_polar(self, log_scaling=True):
        n = len(self.marks)
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = np.log1p(self.marks) if log_scaling else np.array(self.marks)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, r, theta

    def compute_mass_density(self, x, y, size=512, sigma=2.0):
        max_extent = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.1
        grid_range = [[-max_extent, max_extent], [-max_extent, max_extent]]
        H, xedges, yedges = np.histogram2d(x, y, bins=size, range=grid_range)
        return gaussian_filter(H, sigma=sigma), (xedges, yedges)

    def robust_fractal_dimension(self, Z, epsilons=None, threshold_percentile=75):
        """Improved fractal dimension estimation with diagnostics"""
        if epsilons is None:
            max_dim = min(Z.shape)
            epsilons = [2**i for i in range(1, int(np.log2(max_dim)))]
        
        # Handle zero-density case
        if np.max(Z) < 1e-8:
            return np.nan, np.nan, None, None
        
        # Adaptive thresholding
        Z_nonzero = Z[Z > 0]
        if len(Z_nonzero) > 0:
            threshold = np.percentile(Z_nonzero, threshold_percentile)
        else:
            return np.nan, np.nan, None, None
        
        sizes = []
        valid_eps = []
        for eps in epsilons:
            if all(s >= eps for s in Z.shape):
                crop_size = (Z.shape[0] // eps) * eps
                cropped = Z[:crop_size, :crop_size]
                reduced = block_reduce(cropped, block_size=(eps, eps), func=np.mean)
                count = np.sum(reduced > threshold)
                if count > 0:  # Only include scales with occupied boxes
                    sizes.append(count)
                    valid_eps.append(eps)
        
        if len(sizes) < 3:
            return np.nan, np.nan, None, None
        
        logs = np.log(np.array(sizes))
        logs_eps = np.log(1.0 / np.array(valid_eps))
        
        # Linear regression with quality metrics
        coeffs, residuals, _, _, _ = np.polyfit(logs_eps, logs, 1, full=True)
        slope = coeffs[0]
        if len(residuals) > 0:
            ss_res = residuals[0]
            ss_tot = np.sum((logs - np.mean(logs))**2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        else:
            r_squared = 1.0
        
        return slope, r_squared, logs_eps, logs

    def validate_embeddings(self, max_marks=100, num_trials=5):
        """Systematic comparison of embeddings with error analysis"""
        results = {'log': [], 'linear': []}
        r_squareds = {'log': [], 'linear': []}
        
        for _ in tqdm(range(num_trials), desc=f"Testing {max_marks} marks"):
            # Grow new spacetime
            sim = QuantumGolombValidator([0, 1])
            sim.quantum_growth(max_marks=max_marks)
            
            # Test log-polar
            x_log, y_log, _, _ = sim.embed_polar(log_scaling=True)
            Z_log, _ = sim.compute_mass_density(x_log, y_log, size=512)
            dim_log, r2_log, _, _ = sim.robust_fractal_dimension(Z_log)
            if not np.isnan(dim_log):
                results['log'].append(dim_log)
                r_squareds['log'].append(r2_log)
            
            # Test linear-polar
            x_lin, y_lin, _, _ = sim.embed_polar(log_scaling=False)
            Z_lin, _ = sim.compute_mass_density(x_lin, y_lin, size=512)
            dim_lin, r2_lin, _, _ = sim.robust_fractal_dimension(Z_lin)
            if not np.isnan(dim_lin):
                results['linear'].append(dim_lin)
                r_squareds['linear'].append(r2_lin)
        
        # Statistical summary
        summary = {}
        for embed in ['log', 'linear']:
            dims = results[embed]
            r2s = r_squareds[embed]
            if dims:
                summary[embed] = {
                    'mean_dim': np.mean(dims),
                    'std_dim': np.std(dims),
                    'mean_r2': np.mean(r2s),
                    'num_valid': len(dims)
                }
            else:
                summary[embed] = None
        
        return summary

    def visualization_suite(self, max_marks=100):
        """Comprehensive visual validation for a single spacetime"""
        # Grow spacetime
        self.quantum_growth(max_marks=max_marks)
        
        # Create figure
        fig = plt.figure(figsize=(18, 12), tight_layout=True)
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # Configure plots
        ax1 = fig.add_subplot(gs[0, 0])  # Log-polar density
        ax2 = fig.add_subplot(gs[0, 1])  # Linear-polar density
        ax3 = fig.add_subplot(gs[1, 0])  # Log-polar fractal
        ax4 = fig.add_subplot(gs[1, 1])  # Linear-polar fractal
        ax5 = fig.add_subplot(gs[2, :])  # Mark distribution
        
        # Set main title
        fig.suptitle(f'Quantum Golomb Spacetime Validation (n={max_marks})', 
                    fontsize=16, y=0.98)
        
        # 1. Density comparison
        for i, (embed, ax) in enumerate(zip(['log', 'linear'], [ax1, ax2])):
            x, y, _, _ = self.embed_polar(log_scaling=(embed == 'log'))
            Z, (xedges, yedges) = self.compute_mass_density(x, y, size=512)
            
            # Compute fractal dimension
            dim, r2, leps, lN = self.robust_fractal_dimension(Z)
            
            # Plot density
            im = ax.imshow(Z.T, origin='lower', extent=[
                xedges[0], xedges[-1], yedges[0], yedges[-1]
            ], cmap='viridis', aspect='auto')
            
            ax.set_title(f"{'Log' if embed=='log' else 'Linear'}-Polar Embedding\n"
                        f"Fractal dim: {dim:.3f} (R²={r2:.3f})")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            plt.colorbar(im, ax=ax, label='Density')
        
        # 2. Fractal analysis plots
        for i, (embed, ax) in enumerate(zip(['log', 'linear'], [ax3, ax4])):
            x, y, _, _ = self.embed_polar(log_scaling=(embed == 'log'))
            Z, _ = self.compute_mass_density(x, y, size=512)
            dim, r2, logs_eps, logs_N = self.robust_fractal_dimension(Z)
            
            if logs_eps is not None:
                # Plot actual measurements
                ax.scatter(logs_eps, logs_N, c='b', s=50, 
                          label=f'Measurements (dim={dim:.3f})')
                
                # Plot regression line
                fit_line = np.poly1d(np.polyfit(logs_eps, logs_N, 1))
                x_range = np.linspace(min(logs_eps), max(logs_eps), 100)
                ax.plot(x_range, fit_line(x_range), 'r--', 
                       label=f'Fit (R²={r2:.3f})')
                
                ax.set_xlabel('log(1/ε)')
                ax.set_ylabel('log(N(ε))')
                ax.set_title(f"{'Log' if embed=='log' else 'Linear'}-Polar Fractal Analysis")
                ax.legend()
                ax.grid(True)
        
        # 3. Mark distribution analysis
        positions = np.array(self.marks)
        intervals = positions[1:] - positions[:-1]
        cumulative = positions - positions[0]
        
        ax5.plot(positions, 'o-', label='Mark positions')
        ax5.plot(np.arange(1, len(positions)), intervals, 's-', 
                label='Inter-mark distances')
        ax5.plot(cumulative, '^-', label='Cumulative position')
        
        ax5.set_title("Mark Position Analysis")
        ax5.set_xlabel("Mark index")
        ax5.set_ylabel("Position value")
        ax5.legend()
        ax5.grid(True)
        
        # Add informational text
        textstr = '\n'.join([
            f'Total marks: {len(positions)}',
            f'Total span: {positions[-1] - positions[0]}',
            f'Avg interval: {np.mean(intervals):.2f} ± {np.std(intervals):.2f}',
            f'Min interval: {np.min(intervals)}, Max: {np.max(intervals)}'
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, 
                verticalalignment='top', bbox=props)
        
        plt.show()
        
        return fig

    def scaling_analysis(self, mark_range=(20, 401, 20), num_trials=3):
        """Systematic dimension analysis across scales"""
        mark_counts = range(*mark_range)
        results = {n: {'log': [], 'linear': []} for n in mark_counts}
        
        for n in tqdm(mark_counts, desc="Scaling analysis"):
            for _ in range(num_trials):
                sim = QuantumGolombValidator([0, 1])
                sim.quantum_growth(max_marks=n)
                
                # Log-polar
                x_log, y_log, _, _ = sim.embed_polar(log_scaling=True)
                Z_log, _ = sim.compute_mass_density(x_log, y_log, size=512)
                dim_log, _, _, _ = sim.robust_fractal_dimension(Z_log)
                if not np.isnan(dim_log):
                    results[n]['log'].append(dim_log)
                
                # Linear-polar
                x_lin, y_lin, _, _ = sim.embed_polar(log_scaling=False)
                Z_lin, _ = sim.compute_mass_density(x_lin, y_lin, size=512)
                dim_lin, _, _, _ = sim.robust_fractal_dimension(Z_lin)
                if not np.isnan(dim_lin):
                    results[n]['linear'].append(dim_lin)
        
        # Process results
        x_vals = []
        log_means, log_stds = [], []
        lin_means, lin_stds = [], []
        
        for n in mark_counts:
            log_vals = results[n]['log']
            lin_vals = results[n]['linear']
            
            if log_vals:
                x_vals.append(n)
                log_means.append(np.mean(log_vals))
                log_stds.append(np.std(log_vals))
                lin_means.append(np.mean(lin_vals))
                lin_stds.append(np.std(lin_vals))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot with error bars
        plt.errorbar(x_vals, log_means, yerr=log_stds, fmt='o-', 
                    capsize=5, label='Log-polar')
        plt.errorbar(x_vals, lin_means, yerr=lin_stds, fmt='s-', 
                    capsize=5, label='Linear-polar')
        
        # Formatting
        plt.xlabel("Number of Marks")
        plt.ylabel("Fractal Dimension")
        plt.title("Fractal Dimension vs System Size")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Highlight critical region
        if any(m < 1.9 for m in log_means):
            plt.axhspan(1.7, 1.9, color='red', alpha=0.1, 
                        label='Dimensional Reduction')
        
        plt.show()
        
        return results

# =====================
# Validation Workflow
# =====================
if __name__ == "__main__":
    validator = QuantumGolombValidator([0, 1])
    
    # 1. Visual validation for critical sizes
    print("\n=== Visual Validation (40 marks) ===")
    validator.visualization_suite(max_marks=40)
    
    print("\n=== Visual Validation (100 marks) ===")
    validator.visualization_suite(max_marks=100)
    
    print("\n=== Visual Validation (400 marks) ===")
    validator.visualization_suite(max_marks=400)
    
    # 2. Statistical embedding comparison
    print("\n=== Embedding Comparison (100 marks) ===")
    summary_100 = validator.validate_embeddings(max_marks=100, num_trials=10)
    print("\nValidation Summary (100 marks):")
    print("Log-polar: ", summary_100.get('log', 'No valid results'))
    print("Linear-polar: ", summary_100.get('linear', 'No valid results'))
    
    # 3. Full scaling analysis
    print("\n=== Scaling Analysis (20-400 marks) ===")
    scaling_results = validator.scaling_analysis(
        mark_range=(20, 401, 20),  # From 20 to 400 in steps of 20
        num_trials=5
    )
