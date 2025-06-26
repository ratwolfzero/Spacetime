import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from itertools import combinations # Needed for generating pairs of marks
from numba import njit

class DeterministicGolombUniverse:
    """
    A simplified simulation of the Golomb Universe based on a deterministic
    growth rule, as proposed in the paper.
    """
    def __init__(self, initial_marks=None):
        self.marks = sorted(initial_marks) if initial_marks else [0, 1]
        self.distances = self._compute_all_distances(self.marks)
        self.history = [self.marks.copy()]

    def _compute_all_distances(self, marks):
        """Computes all unique pairwise absolute differences for a given set of marks."""
        return {abs(m2 - m1) for m1, m2 in combinations(marks, 2)}

    def _is_valid_addition(self, candidate_mark):
        """
        Checks if adding a candidate mark maintains the Golomb property.
        A candidate is valid if its distances to ALL existing marks are unique
        (i.e., not already in the existing distances set).
        """
        new_dists = {abs(candidate_mark - mark) for mark in self.marks}
        return new_dists.isdisjoint(self.distances)

    def grow(self, max_marks=30):
        """
        Grows the Golomb Universe using the deterministic (greedy) rule:
        Always adds the smallest possible new mark that preserves uniqueness.
        """
        print(f"Starting growth with marks: {self.marks}")
        print(f"Initial distances: {sorted(list(self.distances))}")

        while len(self.marks) < max_marks:
            current_max_mark = self.marks[-1]
            next_mark_found = False
            candidate = current_max_mark + 1

            while not next_mark_found:
                if self._is_valid_addition(candidate):
                    next_mark_found = True
                    break
                candidate += 1

            chosen_mark = candidate
            new_dists_from_chosen = {abs(chosen_mark - mark) for mark in self.marks}

            self.marks.append(chosen_mark)
            self.distances.update(new_dists_from_chosen)

            self.history.append(self.marks.copy())

        print(f"\nGrowth complete. Final {len(self.marks)} marks: {self.marks}")
        print(f"Total unique distances: {len(self.distances)}")
        return self.marks

    def plot_marks_linear(self):
        """Visualizes the Golomb ruler marks on a simple 1D line."""
        plt.figure(figsize=(12, 3))
        plt.scatter(self.marks, np.zeros_like(self.marks), s=100, zorder=5, color='blue', edgecolors='k')
        for i, mark in enumerate(self.marks):
            plt.text(mark, 0.05, str(mark), ha='center', va='bottom', fontsize=9)
        plt.hlines(0, self.marks[0], self.marks[-1], color='gray', linestyle='-', linewidth=1)
        plt.title(f"Deterministic Golomb Universe: Marks (N={len(self.marks)})")
        plt.xlabel("Mark Position")
        plt.yticks([])
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.ylim(-0.1, 0.2)
        plt.show()

    def plot_distances_linear(self):
        """
        Visualizes the marks along with all the unique distances they form.
        Each distance is drawn as a horizontal line connecting two marks,
        with its value labeled above. Lines are vertically offset to avoid overlap.
        """
        if len(self.marks) < 2:
            print("Not enough marks to show distances.")
            return

        plt.figure(figsize=(16, 10)) # Increased figure size for better visibility
        ax = plt.gca()

        y_mark_pos = 0 # Baseline for the marks
        plt.scatter(self.marks, np.full_like(self.marks, y_mark_pos), s=150, zorder=5, color='blue', edgecolors='k', label='Marks') # Slightly larger marks
        for mark in self.marks:
            plt.text(mark, y_mark_pos + 0.12, str(mark), ha='center', va='bottom', fontsize=12, color='darkblue', weight='bold') # Larger, bold labels for marks
        plt.hlines(y_mark_pos, self.marks[0], self.marks[-1], color='gray', linestyle='-', linewidth=1.5, zorder=0) # Thicker baseline


        distance_data = []
        for m1, m2 in combinations(self.marks, 2):
            dist = m2 - m1
            distance_data.append((dist, m1, m2))

        distance_data.sort(key=lambda x: x[0]) # Sort by distance value (shortest to longest)

        max_dist_value = self.marks[-1] - self.marks[0]
        # Adjusted y_scale_factor to provide more vertical separation.
        # This constant (e.g., 3.0 or 4.0) controls how "tall" the distance section is.
        y_scale_factor = 3.0 / max_dist_value if max_dist_value > 0 else 0.1 # Ensures lines spread out vertically

        current_overall_y_offset = y_mark_pos + 0.4 # Start plotting distances a bit higher above the marks

        for dist_val, m1, m2 in distance_data:
            # Determine y-level for this distance's line and text
            y_level_for_this_dist = current_overall_y_offset + (dist_val * y_scale_factor)
            
            # Draw the line representing the distance
            plt.plot([m1, m2], [y_level_for_this_dist, y_level_for_this_dist],
                     'r-', linewidth=2, alpha=1.0) # Thicker, fully opaque red lines

            # Add the distance label
            plt.text((m1 + m2) / 2, y_level_for_this_dist + 0.08, str(dist_val), # Increased text vertical offset
                     ha='center', va='bottom', fontsize=11, color='darkred', weight='bold') # Larger, bold labels

        plt.title(f"Deterministic Golomb Universe: Marks and All Unique Distances (N={len(self.marks)})")
        plt.xlabel("Mark Position")
        plt.yticks([]) # Hide y-axis for cleaner visualization
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Adjust y-limits dynamically to show all marks and distances clearly
        # Increased top padding to ensure labels don't get cut off
        max_y_value_plotted = current_overall_y_offset + (max_dist_value * y_scale_factor)
        plt.ylim(y_mark_pos - 0.2, max_y_value_plotted + 0.5) # Increased top padding
        plt.tight_layout() # Adjust layout to prevent labels/titles from cutting off
        plt.show()

    def plot_polar_embedding(self, log_scaling=True):
        """
        Embeds the marks into a 2D polar coordinate system (spiral).
        This is a key visualization used in your paper to hint at emergent space.
        """
        n = len(self.marks)
        if n == 0:
            print("No marks to plot.")
            return

        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = np.array(self.marks)

        if log_scaling and r.max() > 0:
            r = np.log1p(r)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        plt.figure(figsize=(8, 8))
        sc = plt.scatter(x, y, c=self.marks, cmap='viridis', s=30, edgecolors='k', linewidth=0.2)
        plt.colorbar(sc, label='Temporal Position (Mark Value)')
        plt.title(f"Deterministic Golomb Universe: Polar Embedding (N={n} Marks)")
        plt.xlabel("X (Emergent Space)")
        plt.ylabel("Y (Emergent Space)")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, alpha=0.3)
        plt.show()

# --- Simulation Execution ---
if __name__ == "__main__":
    universe_simulator = DeterministicGolombUniverse(initial_marks=[0, 1])

    # Grow the universe to a moderate number of marks to clearly see distances
    # Too many marks (e.g., > 12-15) will make the distance plot very cluttered
    # I've set it to 10 for optimal visibility of distance lines for now.
    universe_simulator.grow(max_marks=7)

    # Plot the marks linearly
    universe_simulator.plot_marks_linear()

    # Plot the marks with their unique distances (new visualization)
    universe_simulator.plot_distances_linear()

    # Plot the polar embedding
    universe_simulator.plot_polar_embedding(log_scaling=True)

    print("\nGenerated Golomb Ruler Marks:", universe_simulator.marks)
    print("All unique distances (count):", len(universe_simulator.distances))
