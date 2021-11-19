"""
You can create any other helper funtions.
Do not modify the given functions
"""


def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path = []
    # TODO
    return path


def dfs(visited, graph, node, goal_nodes, dfs_path, cost_incurred):
    if node in goal_nodes:
        return cost_incurred, dfs_path

    if node not in visited:
        visited.add(node)
        dfs_path.append(node)
        # print(node, end=' ')
        for i in range(len(graph[0])):
            if graph[node][i] > 0:
                cost_incurred += graph[node][i]
                dfs(visited, graph, i, goal_nodes, dfs_path, cost_incurred)

    print("Cost: {} Path: {}".format(cost_incurred, dfs_path))
    if len(visited) == len(graph[0]) + 1:
        return -1, -1


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []

    for i in range(len(goals)):
        visited = set()
        temp_path = []
        cost_incurred, path_ = dfs(visited, cost, start_point, goals, temp_path, 0)
        print('Cost: {} Path: {}'.format(cost_incurred, path_))
        if path_ != -1:
            goals.remove(path_[-1])
    return path
