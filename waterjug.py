from collections import deque
from math import gcd

def minSteps(m, n, d):
    # Check if the problem is solvable
    if d > max(m, n) or d % gcd(m, n) != 0:
        return "Can't obtain the required state"

    visited = set()
    q = deque([((0, 0), [])])  # Each item: ((jug1, jug2), path to state)

    while q:
        (a, b), path = q.popleft()

        
        if a == d or b == d:
            path.append((a, b))
            print("Steps to reach the target state:")
            for state in path:
                print(state)
            return len(path) - 1

        if (a, b) in visited:
            continue
        visited.add((a, b))

        
        next_states = [
            (m, b),              # Fill jug1
            (a, n),              # Fill jug2
            (0, b),              # Empty jug1
            (a, 0),              # Empty jug2
            (a - min(a, n - b), b + min(a, n - b)),  # Pour jug1 → jug2
            (a + min(b, m - a), b - min(b, m - a))   # Pour jug2 → jug1
        ]

        for state in next_states:
            if state not in visited:
                q.append((state, path + [(a, b)]))

    return "No solution found"

# Example usage
print("Minimum steps:", minSteps(4, 3, 2))
