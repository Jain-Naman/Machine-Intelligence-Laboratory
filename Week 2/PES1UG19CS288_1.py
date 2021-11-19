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


def dfs(start_point, goal_state, visited, cost, stack, path, curr_cost):
    if start_point == goal_state:
        return cost[start_point][stack.pop()]

    # print('Start point: {}, Goal state: {}'.format(start_point, goal_state))
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

    temp_path = []

    for i in goals:
        visited = []
        stack = []
        cost_inc = dfs(start_point, i, visited, cost, stack, temp_path, 0)
        print(cost_inc)
        print(temp_path)

    return path
