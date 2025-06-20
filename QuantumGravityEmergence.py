import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft
import warnings

class QuantumGolombSpacetime:
    def __init__(self, initial_marks=None):
        self.marks = sorted(initial_marks) if initial_marks else [0, 1]
        self.distances = self._compute_distances(self.marks)
        self.history = [self.marks.copy()]
        self.quantum_state = None
        self.energy_levels = []
        self.planck_length = 1.0  # Fundamental quantum unit

    def _compute_distances(self, marks):
        return {abs(a - b) for i, a in enumerate(marks) for b in marks[i+1:]}
    
    def _is_valid_addition(self, candidate):
        new_dists = {abs(candidate - mark) for mark in self.marks}
        return new_dists.isdisjoint(self.distances)
    
    def quantum_growth(self, max_marks=30, temperature=0.02, search_limit=1000):
        while len(self.marks) < max_marks:
            current_max = self.marks[-1]
            
            # Quantum superposition of possible new marks
            candidate_cloud = []
            for offset in range(1, search_limit + 1):
                candidate = current_max + offset
                if self._is_valid_addition(candidate):
                    candidate_cloud.append(candidate)
                if len(candidate_cloud) >= 100:  # Limit candidate pool size
                    break
            
            if not candidate_cloud:
                print(f"Stopped growing at {len(self.marks)} marks")
                break
                
            # Quantum tunneling probability distribution
            positions = np.arange(len(candidate_cloud))
            probabilities = np.exp(-positions/temperature)
            probabilities /= probabilities.sum()
            
            # Quantum measurement process
            chosen = np.random.choice(candidate_cloud, p=probabilities)
            
            # Update quantum state
            new_dists = {abs(chosen - m) for m in self.marks}
            self.distances.update(new_dists)
            self.marks.append(chosen)
            self.history.append(self.marks.copy())
            
            # Quantize the new position
            self.quantum_state = self.quantize_spacetime()
            
        return self.marks

    def quantize_spacetime(self):
        """Convert discrete marks into quantum wavefunction"""
        positions = np.array(self.marks)
        if len(positions) == 0:
            return np.array([0])
        
        # Create a dense grid
        wavefunction = np.zeros(int(positions[-1] * 1.2) + 100)
        
        # Create wave packets at each mark position
        for pos in positions:
            x = np.arange(len(wavefunction))
            # Gaussian wave packets representing quantum events
            wavefunction += np.exp(-(x - pos)**2 / (2*self.planck_length))
        
        # Normalize the quantum state
        norm = np.sqrt(np.sum(wavefunction**2))
        return wavefunction / norm if norm > 0 else wavefunction

    def compute_spectrum(self):
        """Calculate the energy spectrum of spacetime"""
        if self.quantum_state is None or len(self.quantum_state) < 2:
            self.quantize_spacetime()
            if len(self.quantum_state) < 2:
                return np.array([])
        
        # Fourier transform to momentum space
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            energy = np.abs(fft(self.quantum_state))
        
        # Apply Planck-Einstein relation E = ħω
        energy_spectrum = energy * 2 * np.pi  # ħ = 1 in natural units
        
        # Store energy levels
        self.energy_levels = energy_spectrum
        return energy_spectrum

    def analyze_quantization(self):
        """Perform quantum statistical analysis"""
        if not self.energy_levels:
            self.compute_spectrum()
            if len(self.energy_levels) < 3:
                print("Not enough energy levels for analysis")
                return 0, 0
        
        # Level spacing statistics
        sorted_energy = np.sort(self.energy_levels)
        spacings = np.diff(sorted_energy)
        
        # Filter out zero spacings
        spacings = spacings[spacings > 1e-10]
        
        if len(spacings) < 2:
            print("Not enough spacings for analysis")
            return 0, 0
            
        mean_spacing = np.mean(spacings)
        normalized_spacings = spacings / mean_spacing
        
        # Quantum chaos metrics - handle division by zero safely
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_vals = []
            for s in normalized_spacings:
                if s > 1e-10:  # Avoid division by near-zero
                    r_vals.append(min(s, 1/s))
            
            if r_vals:
                r_value = np.mean(r_vals)
            else:
                r_value = 0
                
        spectral_rigidity = self.compute_spectral_rigidity(sorted_energy)
        
        # Plot results
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Wavefunction plot
        axs[0,0].plot(self.quantum_state, 'b-', linewidth=1)
        axs[0,0].set_title('Quantum Spacetime Wavefunction')
        axs[0,0].set_xlabel('Position (Planck units)')
        axs[0,0].set_ylabel('ψ(x)')
        
        # Energy spectrum
        axs[0,1].plot(self.energy_levels, 'r-')
        axs[0,1].set_title('Energy Spectrum')
        axs[0,1].set_xlabel('State index')
        axs[0,1].set_ylabel('Energy')
        
        # Level spacing distribution
        if len(normalized_spacings) > 1:
            axs[1,0].hist(normalized_spacings, bins=30, density=True, alpha=0.6)
            x = np.linspace(0, 3, 100)
            # Poisson distribution (integrable systems)
            axs[1,0].plot(x, np.exp(-x), 'k--', label='Poisson (Classical)')
            # Wigner surmise (quantum chaotic systems)
            axs[1,0].plot(x, (np.pi*x/2)*np.exp(-np.pi*x**2/4), 'r-', label='Wigner (Quantum Chaos)')
            axs[1,0].set_title('Level Spacing Distribution')
            axs[1,0].set_xlabel('Normalized spacing')
            axs[1,0].legend()
        else:
            axs[1,0].axis('off')
            axs[1,0].text(0.5, 0.5, 'Not enough spacings', 
                          ha='center', va='center')
        
        # Information display
        textstr = '\n'.join((
            f'Quantum Events: {len(self.marks)}',
            f'Spacetime Span: {self.marks[-1] - self.marks[0]} Planck units',
            f'R-value: {r_value:.4f} (1.0 = Poisson, 0.53 = GOE)',
            f'Spectral Rigidity: {spectral_rigidity:.4f}',
            f'Min Spacing: {np.min(spacings):.2e}, Max: {np.max(spacings):.2e}'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[1,1].text(0.05, 0.95, textstr, transform=axs[1,1].transAxes, 
                      verticalalignment='top', bbox=props)
        axs[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return r_value, spectral_rigidity

    def compute_spectral_rigidity(self, eigenvalues, L=0.1):
        """Calculate spectral rigidity Δ₃(L) - quantum chaos indicator"""
        N = len(eigenvalues)
        if N < 3:
            return 0
        
        # Unfold spectrum to constant density
        unfolded = np.arange(N)
        
        min_rigidity = float('inf')
        for i in range(N):
            start = i
            end = min(i + int(L*N), N)
            if end - start < 2:
                continue
                
            # Local spectral rigidity
            local_energies = unfolded[start:end]
            n = len(local_energies)
            k = np.arange(n)
            
            # Fit a straight line
            A = np.vstack([k, np.ones(n)]).T
            m, c = np.linalg.lstsq(A, local_energies, rcond=None)[0]
            
            # Compute deviation from linearity
            deviation = np.sum((local_energies - (m*k + c))**2) / n
            if deviation < min_rigidity:
                min_rigidity = deviation
                
        return min_rigidity

    def holographic_projection(self):
        """Project bulk spacetime to boundary CFT"""
        positions = np.array(self.marks)
        if len(positions) < 2:
            print("Not enough marks for projection")
            return np.array([])
        
        max_pos = positions[-1]
        if max_pos == 0:
            max_pos = 1  # Avoid division by zero
        
        # Conformal mapping: z = exp(2πi * position / max_position)
        boundary_points = np.exp(2j * np.pi * positions / max_pos)
        
        # Boundary correlation function
        correlations = []
        for i in range(len(boundary_points)):
            for j in range(i+1, len(boundary_points)):
                dist = np.abs(boundary_points[i] - boundary_points[j])
                correlations.append(dist)
        
        # Plot boundary theory
        plt.figure(figsize=(10, 8))
        plt.scatter(boundary_points.real, boundary_points.imag, 
                   c=positions, cmap='viridis', s=50)
        plt.title('Holographic Boundary Projection')
        plt.xlabel('Re(z)')
        plt.ylabel('Im(z)')
        plt.colorbar(label='Radial Position')
        plt.gca().set_aspect('equal')
        plt.grid(True, alpha=0.3)
        
        # Show correlation distribution
        plt.figure(figsize=(10, 4))
        if correlations:
            plt.hist(correlations, bins=30, alpha=0.7)
            plt.title('Boundary Correlation Distribution')
            plt.xlabel('|z_i - z_j|')
            plt.ylabel('Count')
        else:
            plt.text(0.5, 0.5, 'No correlations to display', 
                     ha='center', va='center')
        plt.show()
        
        return boundary_points

# =====================
# Quantum Spacetime Analysis
# =====================
if __name__ == "__main__":
    # Create and evolve quantum spacetime
    universe = QuantumGolombSpacetime([0, 1])
    universe.quantum_growth(max_marks=40, temperature=0.05)
    
    # Analyze quantum properties
    print("Quantum State Analysis:")
    r_value, rigidity = universe.analyze_quantization()
    print(f"Final R-value: {r_value:.4f}")
    print(f"Spectral Rigidity: {rigidity:.4f}")
    
    # Holographic projection
    print("\nHolographic Boundary Projection:")
    boundary_points = universe.holographic_projection()
    print(f"Generated {len(boundary_points)} boundary points")
