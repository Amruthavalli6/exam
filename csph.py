def is_valid(graph, var, color, assignment):
    return all(assignment.get(neigh) != color for neigh in graph[var])

def backtrack(graph, colors, assignment={}):
    if len(assignment) == len(graph):
        return assignment
    
    unassigned = [v for v in graph if v not in assignment]
    var = unassigned[0]
    
    for color in colors:
        if is_valid(graph, var, color, assignment):
            assignment[var] = color
            result = backtrack(graph, colors, assignment)
            if result:
                return result
            del assignment[var]  # backtrack
    
    return None

# Example use:
result = backtrack({
    'WA': ['NT'],
    'NT': ['WA', 'SA', 'QLD'],
    'SA': ['WA', 'NT', 'QLD', 'VIC'],
    'QLD': ['NT', 'SA', 'NSW'],
    'NSW': ['QLD', 'SA', 'VIC'],
    'VIC': ['SA', 'NSW'],
    'TAS': []
}, colors=['red', 'green', 'blue'])

print(result)
