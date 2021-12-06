"""
Contains visualizations for a_star
"""
from collections import defaultdict
from algorithms.classical_search import a_star, djikstra
import matplotlib.pyplot as plt
import random

random.seed(0)


def print_grid(m, n, blocks=None, explored=None):
    if blocks is None:
        blocks = []

    if explored is None:
        explored = []

    grid = [["." for j in range(n)] for i in range(m)]

    for block in blocks:
        i, j = block
        grid[i][j] = "#"

    for exp in explored:
        i, j = exp
        grid[i][j] = "+"

    for row in grid:
        print(" ".join(row))


def grid_graph(m, n, blocks=None):
    if blocks is None:
        blocks = []

    vertices = [(i, j) for i in range(m) for j in range(n)]

    graph = defaultdict(list)
    for vertex in vertices:
        i, j = vertex
        for child in [
            (i+1, j),
            (i, j+1),
            (i-1, j),
            (i, j-1)
        ]:
            ic, jc = child

            if 0 <= ic < m and 0 <= jc < n and child not in blocks:
                graph[vertex].append(child)

    return graph


def manh_d(start, end):
    return abs(end[0] - start[0]) + abs(end[1] - start[1])


def plt_result(grid, start, end, path, explored):
    nrows, ncols, blocks = grid
    vertices = [(i, j) for i in range(nrows) for j in range(ncols)]
    plt.scatter(*zip(*vertices), marker='.', alpha=0.5)
    plt.scatter(*zip(*explored), marker="+", label="explored", alpha=0.7)
    plt.scatter(*zip(*blocks), c="red")
    plt.scatter(*zip(*path), label="path")
    plt.scatter(*start, label="start")
    plt.scatter(*end, label="end")
    plt.legend()


if __name__ == '__main__':
    GRID1 = (10, 11, [
        (1,4),
        (2,4),
        (3,4),
        (5, 4),
        (6, 4),
        (7, 4),
        (7, 5),
        (7, 6),
        (7, 7)
    ]), (7,2), (5, 7)
    GRID2 = (100, 100, {
        (random.randint(10, 70), random.randint(0, 99))
        for _ in range(3000)
    }), (0, 0), (99, 99)
    GRID3 = (30, 30, [
        (5, 20),
        (6, 20),
        (7, 20),
        (8, 20),
        (9, 20),
        (10, 20),
        (11, 20),
        (12, 20),
        (13, 20),
        (14, 20),
        (15, 20),
        (16, 20),
        (17, 20),
        (17, 19),
        (17, 18),
        (17, 17),
        (17, 16),
        (17, 15),
        (17, 14),
        (17, 13),
        (17, 12),
        (17, 11),
        (17, 10),
        (17, 9),
        (17, 8),
        (17, 7),
        (17, 6),
        (17, 5),
        (17, 4),
        (17, 3)
    ]), (15, 5), (15, 25)

    for grid, start, end in [GRID3]:
        print("Djikstra")
        graph = grid_graph(*grid)
        path, explored = djikstra(graph, start=start, end=end, cost=manh_d)
        print_grid(*grid, explored=explored)
        print()

        plt_result(grid, start, end, path, explored)
        plt.title("Djikstra's Algorithm")
        plt.savefig("images/gridworld_djikstra.png")
        plt.show()

        print("A_star")
        graph = grid_graph(*grid)
        path, explored = a_star(graph, start=start, end=end, cost=manh_d, heuristic=manh_d)
        print_grid(*grid, explored=explored)
        print("\n\n")

        plt_result(grid, start, end, path, explored)
        plt.title("A* Algorithm")
        plt.savefig("images/gridworld_a_star.png")
        plt.show()


