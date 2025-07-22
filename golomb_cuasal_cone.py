import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import random
import numpy as np

# --- Golomb Growth Rule (Robustified with clear message) ---
def golomb_grow(current_ruler_elements: np.ndarray, num_to_add: int, max_m_search: int = 50000) -> np.ndarray:
    """
    Generates the next 'num_to_add' elements for a Golomb ruler,
    extending from the 'current_ruler_elements'.
    Includes a 'max_m_search' parameter to prevent infinite loops and
    prints a warning if the search limit is hit.
    """
    if num_to_add <= 0:
        return np.array([], dtype=np.int64)

    all_diffs_set = set()
    for i in range(len(current_ruler_elements)):
        for j in range(i + 1, len(current_ruler_elements)):
            all_diffs_set.add(abs(current_ruler_elements[j] - current_ruler_elements[i]))

    max_current_val = 0
    if len(current_ruler_elements) > 0:
        max_current_val = np.max(current_ruler_elements)

    next_elements = np.zeros(num_to_add, dtype=np.int64)
    elements_found = 0

    m = max_current_val + 1 if max_current_val > 0 else 1

    m_search_counter = 0

    while elements_found < num_to_add:
        # Check for search limit BEFORE trying to find a new m
        if m_search_counter >= max_m_search:
            print(f"\n--- WARNING: Golomb search limit reached! ---")
            print(f"    Failed to find {num_to_add - elements_found} additional Golomb elements.")
            print(f"    Current ruler length: {len(current_ruler_elements)}")
            print(f"    Last 'm' attempted: {m}")
            print(f"    Returning {elements_found} elements found so far. Simulation will continue.")
            print(f"---------------------------------------------")
            break # Exit the while loop if search limit is hit

        valid_m = True
        temp_new_diffs_set = set()

        for i in range(len(current_ruler_elements)):
            diff = m - current_ruler_elements[i]
            if diff <= 0:
                valid_m = False
                break

            if diff in all_diffs_set:
                valid_m = False
                break
							
            if diff in temp_new_diffs_set:
                valid_m = False
                break
            temp_new_diffs_set.add(diff)

        if valid_m:
            all_diffs_set.update(temp_new_diffs_set)

            next_elements[elements_found] = m
            elements_found += 1
            max_current_val = m
            m_search_counter = 0 # Reset counter after finding a valid element
            m += 1
        else:               
            m += 1
            m_search_counter += 1 # Increment counter only if m increases without finding a solution

    return next_elements[:elements_found] # Return only the elements found


# -------------------------------
# Parameters
# -------------------------------
num_levels = 50      # Number of generative time steps - Testing with 30
max_new_distinctions_per_parent = 2 # Max children per parent, allows more branching
modal_chance = 0.3  # Chance a distinction remains modal (ghost)
# random.seed(42)     # Commented out for truly random behavior each run

# -------------------------------
# Initialize Graph
# -------------------------------
G = nx.DiGraph()
G.add_node(0, level=0, energy=0.0, realized=True, golomb_value=0) # Axiom 0 — the Void
node_id = 1           
nodes_by_level = {0: [0]}
history = []

all_golomb_values_master_set = {0}


# -------------------------------
# Generate Causal Structure using Golomb Growth
# -------------------------------
for level in range(1, num_levels):
    nodes_by_level[level] = []

    current_ruler_elements_for_grow_func = np.array(sorted(list(all_golomb_values_master_set)), dtype=np.int64)

    num_parents_at_prev_level = len(nodes_by_level[level - 1])

    desired_num_to_add = random.randint(1, max_new_distinctions_per_parent * max(1, num_parents_at_prev_level))

    # CRITICAL CAP: Adjust this value to balance density vs. performance/stability.
    num_to_add_capped = min(desired_num_to_add, 5) # Capped at 5 new elements per call
					  
    if num_to_add_capped == 0:
        continue

    new_golomb_values_candidates = golomb_grow(current_ruler_elements_for_grow_func, num_to_add_capped)

    actual_new_golomb_values = []
    # FIX IS HERE: Corrected the 'for val in val in ...' line
    for val in new_golomb_values_candidates:
        if val not in all_golomb_values_master_set:
            actual_new_golomb_values.append(val)

    if not actual_new_golomb_values: # If no truly new Golomb values were found (e.g., if cap was too low)
        continue

    for val in actual_new_golomb_values:
        all_golomb_values_master_set.add(val)

    parent_index = 0
    num_parents = len(nodes_by_level[level - 1])

    if num_parents == 0 and level > 0:
        continue

    for golomb_val in actual_new_golomb_values:
        parent_node = nodes_by_level[level - 1][parent_index % num_parents] if num_parents > 0 else 0

        energy = G.nodes[parent_node]['energy'] + golomb_val / (num_levels * 5.0) + random.uniform(0.1, 0.5)
        realized = random.random() > modal_chance

        G.add_node(node_id, level=level, energy=energy, realized=realized, golomb_value=golomb_val)
        G.add_edge(parent_node, node_id)
        nodes_by_level[level].append(node_id)
        history.append((parent_node, node_id))
        node_id += 1

        parent_index += 1


# -------------------------------
# Layout Positions (adjusted for Golomb values for a cone shape)
# -------------------------------
pos = {}
for level in range(num_levels):
    level_nodes = nodes_by_level.get(level, [])
    if not level_nodes:
        continue

    golomb_vals_at_level = [G.nodes[n]['golomb_value'] for n in level_nodes]

    if not golomb_vals_at_level:
        continue
					  
    min_golomb_val = min(golomb_vals_at_level)
    max_golomb_val = max(golomb_vals_at_level)

    level_spread_width = level * 2.0

    for node in level_nodes:
        golomb_val = G.nodes[node]['golomb_value']

        if min_golomb_val == max_golomb_val:
            normalized_x = 0.5
        else:
            normalized_x = (golomb_val - min_golomb_val) / (max_golomb_val - min_golomb_val)

        x_pos = (normalized_x - 0.5) * level_spread_width
        pos[node] = (x_pos, -level)


# -------------------------------
# Animation Setup
# -------------------------------
fig, ax = plt.subplots(figsize=(14, 10))

initial_min_e, initial_max_e = 0, 1
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=initial_min_e, vmax=initial_max_e))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Distinction Energy (Axiom V)', fontsize=10)


def update(frame):
    ax.clear()
    step_edges = history[:frame + 1]
    step_nodes = set([src for src, _ in step_edges] + [tgt for _, tgt in step_edges])

    realized_nodes_in_frame = [n for n in step_nodes if G.nodes[n]['realized']]
    energies = [G.nodes[n]['energy'] for n in realized_nodes_in_frame]

    if energies:
        min_e = min(energies)
        max_e = max(energies)
    else:
        min_e, max_e = 0, 1

    if min_e == max_e:
        max_e = min_e + 1e-5

    node_colors = []
    node_labels = {}
    node_list_to_draw = []

    for node in step_nodes:
        data = G.nodes[node]
        node_list_to_draw.append(node)
        if data['realized']:
            norm_energy = (data['energy'] - min_e) / (max_e - min_e)
            color = plt.cm.viridis(norm_energy)
            label = str(node)
        else:
            color = (0.7, 0.7, 0.7, 0.3)
            label = f"◦{node}"
        node_colors.append(color)
        node_labels[node] = label

    if node_list_to_draw:
        nx.draw_networkx_nodes(G, pos, nodelist=node_list_to_draw, node_color=node_colors, node_size=300, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)

    nx.draw_networkx_edges(G, pos, edgelist=step_edges, arrows=True, alpha=0.4, ax=ax)

    ax.set_title(f"Causal Cone with Golomb-Inspired Distinctions — Step {frame + 1}", fontsize=14)
    ax.axis('off')     

    sm.set_norm(plt.Normalize(vmin=min_e, vmax=max_e))
    cbar.update_normal(sm)

    ax.text(1.02, 1.0, "Node Types:", transform=ax.transAxes, fontsize=10)
    ax.text(1.02, 0.95, "● n : Realized", transform=ax.transAxes, fontsize=9)
    ax.text(1.02, 0.90, "◦n : Modal (unrealized)", transform=ax.transAxes, fontsize=9)


ani = animation.FuncAnimation(fig, update, frames=len(history), interval=80, repeat=False)

# -------------------------------
# To Save as MP4 (Optional)
# -------------------------------
# ani.save("causal_cone.mp4", writer='ffmpeg', fps=10)

# -------------------------------
# Show Animation
# -------------------------------
plt.show()
