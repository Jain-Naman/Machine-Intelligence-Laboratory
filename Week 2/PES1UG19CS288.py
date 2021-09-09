"""
You can create any other helper funtions.
Do not modify the given functions
"""


def get_index(path_cost_map, new_path):
    for j, i in enumerate(path_cost_map):
        if i[0] == new_path:
            return j


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
    alternate = [start_point]
    path_cost_map = [(alternate, heuristic[start_point])]

    while len(path_cost_map) > 0:
        curr_path, curr_cost = path_cost_map[0]
        node = curr_path[-1]
        curr_cost = curr_cost - heuristic[node]
        path_cost_map = path_cost_map[1:]
        path.append(node)
        child_nodes = []

        for i in range(len(cost[0])):
            if cost[node][i] > 0:
                child_nodes.append(i)

        if node in goals:
            return curr_path

        for c_node in child_nodes:
            existing_paths = [i[0] for i in path_cost_map]
            new_path = curr_path + [c_node]
            new_path_cost = curr_cost + cost[node][c_node] + heuristic[c_node]

            if new_path not in existing_paths and c_node not in alternate:
                path_cost_map.append((new_path, new_path_cost))
            elif new_path in existing_paths:
                index = get_index(path_cost_map, new_path)
                path_cost_map[index][1] = min(path_cost_map[index][1], new_path_cost)

            path_cost_map = sorted(path_cost_map, key=lambda value: (value[1], value[0]))
    return path


def dfs(start_point, goal_state, visited, cost, stack, path, curr_cost):
    if start_point == goal_state:
        return cost[start_point][stack.pop()]

    stack.append(start_point)
    visited.append(start_point)
    for i in range(1, len(cost[start_point])):
        if i not in visited:
            visited.append(i)
            if cost[start_point][i] > 0:
                stack.pop()
                stack.append(i)
                path.append(i)
                dfs(i, goal_state, visited, cost, stack, path, curr_cost)
                curr_cost += cost[start_point][i]
    return curr_cost


def DFS_Traversal(cost, start_point, goals):
    """
        Perform DFS Traversal and find the optimal path
            cost: cost matrix (list of floats/int)
            start_point: Staring node (int)
            goals: Goal states (list of ints)
        Returns:
            path: path to goal state obtained from DFS (list of ints)
        """
    path = []

    cost_list = []
    for i in goals:
        visited = []
        temp_path = [start_point]
        stack = []
        cost_inc = dfs(start_point, i, visited, cost, stack, temp_path, 0)
        cost_list.append(cost_inc)
        path.append(temp_path)

    index = cost_list.index(min(cost_list))
    return path[index]
