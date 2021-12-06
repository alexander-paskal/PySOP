"""
Contains visualizations for minimax
"""

from algorithms.classical_search import minimax
import random


random.seed(1)


def gridgame(m, n, minval=0, maxval=9):
    return [[random.randint(minval, maxval) for j in range(n)] for i in range(m)]


def gridgame_graph(m, n):
    graph = {}
    initial = None, None
    initial_children = []
    graph[initial] = initial_children
    for i in range(m):
        children = []
        graph[(i, None)] = children
        initial_children.append((i, None))
        for j in range(n):
            graph[(i, j)] = []
            children.append((i, j))
    return graph


def gridgame_reward(grid):
    def reward(child):
        i, j = child
        return grid[i][j]
    return reward


def print_gridgame(grid):
    grid = [list(map(str, row)) for row in grid]
    rows = [" | ".join(row) for row in grid]
    print(rows[0])
    for row in rows[1:]:
        print("-" * len(row))
        print(row)


if __name__ == '__main__':
    M, N = 5,5

    grid = gridgame(M, N)
    print_gridgame(grid)
    graph = gridgame_graph(M, N)
    reward = gridgame_reward(grid)
    actions = minimax(graph, state=(None, None), depth=3, reward=reward, maxplayer=True)
    for k, v in actions.items():
        print(k, v)