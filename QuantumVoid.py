import random
import math
import statistics

def generate_causal_edges(distinctions_list):
    """
    Generates causal edges between distinctions based on Axiom III (temporal ordering).
    Edges represent morphisms from earlier to later distinctions in the sequence.
    """
    edges = []
    n = len(distinctions_list)
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j)) 
    return edges                                  

# --- QuantumVoid Simulation Class ---                                            
class QuantumVoid:        
    """
    Implements a computational model based on an axiomatic framework of distinctions.

    Axiom 0: Starts with an initial 'void' (single distinction at 0).
    Axiom I: New distinctions introduce unique differences.
    Axiom II: Energy increases based on new unique differences and distinction count.
    """
    def __init__(self, axiomatic_energy_unit=1.0):
        self.distinctions = [0]
        self.raw_energy = 1.0
        self.used_differences = set()
        self.raw_energy_gains = []                     
        self.axiomatic_energy_unit = axiomatic_energy_unit

    def add_distinction(self, candidate_value):
        """
        Attempts to add a new distinction based on Axiom I (unique differences)
        and updates raw energy based on Axiom II.

        Args:
            candidate_value (int): The numerical value of the distinction to attempt to add.

        Returns:
            bool: True if the distinction was successfully added, False otherwise.
        """
        new_diffs = {abs(candidate_value - d) for d in self.distinctions}

        if not new_diffs or (new_diffs & self.used_differences):
            return False

        self.distinctions.append(candidate_value)
        self.used_differences.update(new_diffs)
        
        base_gain = len(new_diffs) * math.log(len(self.distinctions) + 2) * self.axiomatic_energy_unit
        
        energy_gain = base_gain # Energy gain is now simply base_gain
        self.raw_energy += energy_gain
        self.raw_energy_gains.append(energy_gain)

        return True

# --- Simulation Execution Function ---
																								  
def run_single_simulation(axiomatic_energy_unit=1.0,
                          search_strategy='smallest_valid_increment',
                          max_candidate_search_range=200,
                          target_alpha_inv=137.035999084):
    """
    Runs a single simulation of the QuantumVoid model with a given axiomatic energy unit
    and a specified distinction generation strategy, stopping based on a new axiomatic rule.
    """
    universe = QuantumVoid(axiomatic_energy_unit=axiomatic_energy_unit)
																	 
    random_base_step_increment = 5 
    
    # Define the conceptual target for stopping based on your proposal:
    # (n choose 2) + 1 = 137, which occurs at n=17.
    conceptual_alpha_limit_for_distinctions = 137 

    while True: # Loop indefinitely until an internal stopping condition is met
        current_num_distinctions = len(universe.distinctions)
        
        # Check if the *current* system size (in terms of distinctions)
        # has reached or exceeded the conceptual limit for 'maximal stable differences'.
        # This condition means: Stop *after* the Nth distinction has been successfully added,           
        # if that Nth distinction brings the conceptual_diff_count to or above the limit.
        if current_num_distinctions > 0: # Only check after at least the initial void is there
            # Calculate the conceptual value (n choose 2) + 1 for the current number of distinctions
            conceptual_diff_count = (current_num_distinctions * (current_num_distinctions - 1)) / 2 + 1 
            
            # If the conceptual count for the CURRENT number of distinctions meets or exceeds
            # the integer conceptual alpha inverse (137), then we stop.
            # This is the "Principle of Maximal Stable Differences" in action.
            # Added a minimum distinction check to prevent stopping too early if conceptual_alpha_limit is very small.
            if conceptual_diff_count >= conceptual_alpha_limit_for_distinctions and current_num_distinctions >= 17:
                break # System has reached its maximal stable distinction count based on alpha_inv

        # If the system hasn't reached its conceptual limit yet, proceed to try and add another distinction
        last_distinction_value = max(universe.distinctions)
        candidate_added = False                            

        if search_strategy == 'smallest_valid_increment':
            found_candidate_in_search = False
            for increment in range(1, max_candidate_search_range + 1):
                candidate_value = last_distinction_value + increment
                if universe.add_distinction(candidate_value):
                    found_candidate_in_search = True
                    break # Found the smallest valid increment, move to next distinction
            
            if not found_candidate_in_search:
                # If no distinction can be added even with the smallest_valid_increment rule,
                # then the system has truly saturated due to unique difference constraints,             
                # independent of the alpha_inv conceptual limit. This is also a valid stop.
                # In the deterministic smallest_valid_increment case, this typically only happens
                # if the search range is too small or a true mathematical impasse is hit.
                break 
            
        elif search_strategy == 'random_constrained':
            # This branch is less relevant for the emergent stopping condition based on N.
            # It will still try to add random distinctions.
            candidate_value = last_distinction_value + random.randint(
                random_base_step_increment, random_base_step_increment + 10 + len(universe.distinctions) // 3
            )
            if not universe.add_distinction(candidate_value):
                # For random_constrained, if adding fails, we just loop again and try a new random candidate.
                pass
            else:
                # Only increment step if a distinction was successfully added
                random_base_step_increment = min(30, random_base_step_increment + 1)
                
    # Calculate the derived scale factor as before
    scale_factor = target_alpha_inv / universe.raw_energy if universe.raw_energy > 0 else float('inf')
    normalized_alpha_inv = universe.raw_energy * scale_factor                                           

    return {
        "raw_energy": universe.raw_energy,
        "derived_scale_factor": scale_factor,
        "normalized_alpha_inv": normalized_alpha_inv,
        "num_distinctions": len(universe.distinctions),
        "causal_edges": generate_causal_edges(universe.distinctions),
        "scaled_energy_contributions": [f"{e * scale_factor:.3f}" for e in universe.raw_energy_gains]
    }

# --- Multiple Simulation Runner ---
def run_multiple_simulations(num_runs=1000, axiomatic_energy_unit=1.0, 
                             search_strategy='smallest_valid_increment',
                             target_alpha_inv=137.035999084):
    """                                 
    Runs multiple simulations with specified parameters and collects statistical data.
    """
    all_raw_energies = []
    all_derived_scale_factors = []
    all_normalized_alphas = []
    all_num_distinctions = []
                                                                                                
    print("--- Running Multiple QuantumVoid Simulations ---")
    print(f"Axiomatic Energy Unit Used: {axiomatic_energy_unit}")
    print(f"Distinction Search Strategy: {search_strategy}")
    print(f"Normalization Target (α⁻¹): {target_alpha_inv}")

    for i in range(num_runs):
        result = run_single_simulation(axiomatic_energy_unit=axiomatic_energy_unit,
                                       search_strategy=search_strategy,
                                       target_alpha_inv=target_alpha_inv)
        
        all_raw_energies.append(result["raw_energy"])
        all_derived_scale_factors.append(result["derived_scale_factor"])
        all_normalized_alphas.append(result["normalized_alpha_inv"])
        all_num_distinctions.append(result["num_distinctions"])

        # Progress update, prints every 10% of runs or at the very end    
        if (i + 1) % (num_runs // 10 if num_runs >= 10 else 1) == 0 or i == num_runs - 1:
            print(f"Run {i+1}/{num_runs}: Raw Energy = {result['raw_energy']:.3f}, "
                  f"Derived Scale Factor = {result['derived_scale_factor']:.6f}, "
                  f"Num Distinctions = {result['num_distinctions']}")
                  
    print("\n=== Summary Over Multiple Runs ===")
    print(f"Axiomatic Energy Unit Tested: {axiomatic_energy_unit}")
    print(f"Distinction Search Strategy: {search_strategy}")
    print(f"Raw Energy: Mean = {statistics.mean(all_raw_energies):.3f}, "
          f"Std Dev = {statistics.stdev(all_raw_energies):.3f}")
    print(f"Derived Scale Factor: Mean = {statistics.mean(all_derived_scale_factors):.6f}, "
          f"Std Dev = {statistics.stdev(all_derived_scale_factors):.6f}")
    print(f"Normalized α⁻¹ (Target): Mean = {statistics.mean(all_normalized_alphas):.3f}, "
          f"Std Dev = {statistics.stdev(all_normalized_alphas):.3f} (By Construction)")
    print(f"Number of Distinctions: Mean = {statistics.mean(all_num_distinctions):.2f}, "
          f"Min = {min(all_num_distinctions)}, Max = {max(all_num_distinctions)}")

# --- Main Execution ---

if __name__ == "__main__":
    # Parameters for the main test run
    test_params = {
        'num_runs': 5,
        'axiomatic_energy_unit': 0.38607, # Your previously optimized AEU
        'search_strategy': 'smallest_valid_increment',
        'target_alpha_inv': 137.035999084
    }

    # Dynamic title for the run
    print(f"--- Testing '{test_params['search_strategy']}' Strategy (AEU={test_params['axiomatic_energy_unit']}) with Emergent Stopping based on (N choose 2) + 1 ---")
    print("\n" + "="*70 + "\n")

    

    # Run the simulation with the defined parameters                            
    run_multiple_simulations(**test_params)
    
    print("\n" + "="*70 + "\n")

    
