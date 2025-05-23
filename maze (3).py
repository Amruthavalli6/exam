import heapq
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_list = [(0 + heuristic(start, goal), 0, start, [])]  # (f, g, current, path)

    visited = set()

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        if current in visited:
            continue
        visited.add(current)

        path = path + [current]

        if current == goal:
            return path

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            x, y = current[0] + dx, current[1] + dy
            neighbor = (x, y)

            if 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0:
                heapq.heappush(open_list, (g + 1 + heuristic(neighbor, goal), g + 1, neighbor, path))

    return None
grid = [
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 0],
    [1, 1, 0, 0]
]

start = (0, 0)
goal = (3, 2)

path = a_star(grid, start, goal)

print("Path:" if path else "No path found!", path)
