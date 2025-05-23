Graph_nodes = {
    'A': [('B', 2)] , 'B': [('C', 3), ('D', 2)], 'C': [('D', 1)],
    'D': [('C', 1), ('E', 8)], 'E': [('I', 5), ('J', 5)], 'F': [('G', 1), ('H', 7)],
    'G': [('I', 3)], 'H': [('I', 2)], 'I': [('E', 5), ('J', 3)]
}
h = {'A': 10, 'B': 4, 'C': 5, 'D': 7, 'E': 3, 'F': 6, 'G': 5, 'H': 3, 'I': 1, 'J': 0}
def aStar(start, goal):
    open_set, closed_set, g, parents = {start}, set(), {start: 0}, {start: start}

    while open_set:
        n = min(open_set, key=lambda v: g[v] + h[v])
        if n == goal:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            return [start] + path[::-1]

        open_set.remove(n)
        closed_set.add(n)

        for m, weight in Graph_nodes.get(n, []):
            if m in closed_set: continue
            if m not in open_set or g[m] > g[n] + weight:
                g[m] = g[n] + weight
                parents[m] = n
                open_set.add(m)
    return None
print(aStar('A', 'J'))          