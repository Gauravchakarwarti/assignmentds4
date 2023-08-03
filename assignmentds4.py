# 1 Breadth First Traversal for a Graph.

# Solution

from collections import deque
 
def bfs (graph, start):
    queue = deque([start])
    visited = set()
    visited.add(start)

    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C'],
    'C': ['A', 'B', 'D'],
    'D': ['C']

    
}

print("Breadth-First Traversal:")
bfs(graph, 'A')


# 2 Depth First Traversal for a Graph.

# Solution:

def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start, end=' ')

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

graph = {
    'A' : ['B', 'C'],
    'B' : ['A', 'C'],
    'C' : ['A', 'B', 'D'],
    'D' : ['C']

}

print("Depth First Traversal:")
dfs(graph, 'A')


# 3 Count the number of nodes at given level in a tree using BFS.

# Solution:

from collections import deque

def count_nodes_at_level(tree, root, level):
    if not tree or root not in tree:
        return 0

    queue = deque([(root, 0)])
    count = 0
    while queue:
        node, current_level = queue.popleft()

        if current_level == level:
            count += 1
        
        if current_level > level:
            break

        for child in tree[node]:
            queue.append((child, current_level + 1))

    return count


tree = {
    'A' : ['B', 'C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : [],
    'F' : []

}

level = 2
print(f"Number of nodes at level {level}: {count_nodes_at_level(tree, 'A', level)}")


# 4 Count number of trees in a forest.

# Solution:

def count_trees_in_forest(graph):
    def dfs(node, visited):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, visited)

    num_tree = 0
    visited = set()

    for node in graph:
        if node not in visited:
            num_tree += 1
            dfs(node, visited)
    return num_tree

forest = {
    'A' : ['B', 'C'],
    'B' : ['A', 'C'],
    'C' : ['A', 'B'],
    'D' : ['E'],
    'E' : ['D']

}

print(f"Number of tree in the forest: {count_trees_in_forest(forest)}")



# 5 Detect Cycle in a Directed Graph.

# Solution:

def has_cycle(graph):
    def dfs(node, visited, current_path):
        visited.add(node)
        current_path.add(node)

        for neighbor in graph[node]:
            if neighbor in current_path:
                return True
            if neighbor not in visited and dfs(neighbor, visited, current_path):
                return True

        current_path.remove(node)
        return False

    visited = set()
    for node in graph:
        if node not in visited and dfs(node, visited, set()):
            return True

    return False


graph_with_cycle = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['E'],
    'D': ['A'],
    'E': ['B']
}

graph_without_cycle = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['E'],
    'D': ['E'],
    'E': []
}

print("Graph with cycle:", has_cycle(graph_with_cycle))
print("Graph without cycle:", has_cycle(graph_without_cycle))



# Miscellaneous

# 1 Implement n-Queen’s Problem

# Solution:

def is_safe(board, row, col, n):
    for i in range(row):
        if board[i][col] == 1:
            return False

    
    for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
        if board[i][j] == 1:
            return False

    
    for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens_util(board, row, n, solutions):
    if row == n:
       
        solutions.append([''.join('Q' if col == 1 else '.' for col in row) for row in board])
        return

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            solve_n_queens_util(board, row + 1, n, solutions)
            board[row][col] = 0 

def solve_n_queens(n):
    board = [[0 for _ in range(n)] for _ in range(n)]
    solutions = []
    solve_n_queens_util(board, 0, n, solutions)
    return solutions


N = 4
solutions = solve_n_queens(N)

for i, solution in enumerate(solutions):
    print(f"Solution {i+1}:")
    for row in solution:
        print(row)
    print()
