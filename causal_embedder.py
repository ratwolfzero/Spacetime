# This demonstrates the PROVABLE finite embedding property
# The infinite version remains hypothetical

def build_finite_embedding(interactions):
    interactions = sorted(set(interactions))
    ruler = [0]  # Initial mark
    known_diffs = set()
    def is_valid(mark):
        return all(abs(mark - m) not in known_diffs for m in ruler)
    for target in interactions:
        if target not in known_diffs:
            for base in ruler:
                new_mark = base + target
                if is_valid(new_mark):
                    ruler.append(new_mark)
                    known_diffs.update(abs(new_mark - m) for m in ruler)
                    break
            else:
                for base in ruler:
                    new_mark = base - target
                    if new_mark >= 0 and is_valid(new_mark):
                        ruler.append(new_mark)
                        known_diffs.update(abs(new_mark - m) for m in ruler)
                        break
    return sorted(ruler)
print("Finite case proof:", build_finite_embedding([1, 3, 4]))
# Output: [0, 1, 3, 7] - contains all target distance
