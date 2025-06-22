# Demonstrates embedding principle for small M:
# Like local physics preserving information through structural growth
def build_finite_embedding(interactions):
    interactions = sorted(set(interactions))
    ruler = [0]  # Initial mark
    known_diffs = set()  # Track existing differences to enforce uniqueness
    
    # Checks if adding 'mark' preserves Golomb property
    def is_valid(mark):
        return all(abs(mark - m) not in known_diffs for m in ruler)
    
    # Try adding/subtracting target to existing marks
    for target in interactions:
        if target not in known_diffs:
            for base in ruler:
                for new_mark in [base + target, base - target]:  # Test both directions
                    if new_mark >= 0 and is_valid(new_mark):
                        ruler.append(new_mark)
                        known_diffs.update(abs(new_mark - m) for m in ruler)
                        break
    return sorted(ruler)

print("Finite case proof:", build_finite_embedding([1, 3, 4])) 
# Output: [0, 1, 3, 7] (contains all target distances)
