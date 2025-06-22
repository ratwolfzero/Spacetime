import matplotlib.pyplot as plt
import itertools

# Golomb ruler of length 4
marks = [0, 1, 3, 7]
length = marks[-1]

# Plot settings
fig, ax = plt.subplots(figsize=(10, 2))
ax.set_xlim(-1, length + 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Draw the main line
ax.hlines(y=0.5, xmin=0, xmax=length, color='black', linewidth=2)

# Plot ticks and labels
for x in marks:
    ax.vlines(x, 0.3, 0.7, color='black', linewidth=2)
    ax.text(x, 0.75, f"{x}", ha='center', fontsize=12, fontweight='bold')

# Optionally: draw unique distance labels above the line
pairs = list(itertools.combinations(marks, 2))
distances = {abs(j - i): (i, j) for i, j in pairs}  # use a dict to avoid duplicates
for d, (i, j) in sorted(distances.items()):
    xm = (i + j) / 2
    ax.text(xm, 0.95, f"{d}", ha='center', fontsize=10, color='blue')

# Save the figure
plt.tight_layout()
plt.savefig("golomb_ruler_4.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()
