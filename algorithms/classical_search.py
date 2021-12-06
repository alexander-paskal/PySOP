"""
Algorithms for performing classical search on a graph

The graph structure expected for these algorithms is a dictionary mapping
serialized states to lists of other serialized states:

i.e.

    {
        "a": ["b", "c"],
        "b": ["c", "a"],
        "c": ["b", "d"],
        "d": []
    }

All other information i.e. costs, heuristics, etc. are expected to be
functions
"""

import math



def djikstra(graph, start, end, cost):
    """
    Performs djikstra's algorithm to find shortest path from start to end in some graph
    :param graph: a Dict mapping states to Lists of children
    :param start: the state at which to start
    :param end: the state at which to end
    :param cost: a function(start, state) returning the cost from start to state
    :return: path, a List of states from start to finish
    :return: explored, a Set of states that have been visited by the algorithm
    """
    frontier = Queue([start], key=lambda x: cost(start, x))
    explored = set()
    solution = [start]
    parents = {}

    while True:
        if len(frontier) == 0:
            raise RuntimeError("encountered empty frontier queue")
        state = frontier.pop()
        if state == end:
            return _reconstruct_path(start, end, parents), explored
        explored.add(state)
        for child in graph[state]:
            if child not in explored and child not in frontier:
                frontier.append(child)
                parents[child] = state
            elif child in frontier:
                ix = frontier.index(child)
                f_child = frontier[ix]
                f_value = cost(start, f_child)
                n_value = cost(start, child)
                if f_value > n_value:
                    frontier.pop(ix)
                    frontier.append(child)
                    solution[-1] = child
                    parents[child] = state


def a_star(graph, start, end, cost, heuristic):
    """
    Performs A* search from start to end state to find the optimal route. Variation of
    Djikstra's algorithm that employs use of a cost AND a heuristic function so choose which value
    to explore next
    :param graph: a Dict mapping states to Lists of children
    :param start: the state at which to start
    :param end: the state at which to end
    :param cost: a function(state, state) returning the cost of between two states
    :param heuristic: a function(state, state) for calculating the heuristic value between two states
    :return: path, a List of states from start to finish
    :return: explored, a Set of states that have been visited by the algorithm
    """

    frontier = Queue([start], key=lambda x: cost(start, x) + heuristic(x, end))
    explored = set()
    solution = [start]
    parents = {}

    while True:
        if len(frontier) == 0:
            raise RuntimeError("encountered empty frontier queue")
        state = frontier.pop()
        if state == end:
            return _reconstruct_path(start, end, parents), explored
        explored.add(state)
        for child in graph[state]:
            if child not in explored and child not in frontier:
                frontier.append(child)
                parents[child] = state
            elif child in frontier:
                ix = frontier.index(child)
                f_child = frontier[ix]
                f_value = cost(start, f_child) + heuristic(f_child, end)
                n_value = cost(start, child) + heuristic(child, end)
                if f_value > n_value:
                    frontier.pop(ix)
                    frontier.append(child)
                    solution[-1] = child
                    parents[child] = state


def _reconstruct_path(start, end, parents):
    state = end
    path = [end]
    while True:

        if state == start:
            return path[::-1]

        state = parents[state]
        path.append(state)


def minimax(graph, state, depth, reward, maxplayer=True):
    """
    Performs minimax on a 2-player adversarial graph. Returns the values of each of the possible actions from
    the provided state
    :param graph: a Dict mapping states to Lists of children
    :param state: a hashable representation of a state
    :param depth: an Integer for the depth of search to perform
    :param reward: a Function(state) that returns some numeric value
    :return: results, the a Dict mapping children states to their respective values. The "optimal" policy at the provided state would correspond to the argmax of results
    """
    results = {}
    for child in graph[state]:
        results[child] = _minimax(child, graph, reward, depth, maxplayer= not maxplayer)
    return results


def _minimax(state, graph, reward, depth, maxplayer=True):

    if depth == 0:  # max search depth
        return reward(state)

    elif len(graph[state]) == 0:  # leaf state
        return reward(state)

    elif maxplayer:
        value = -math.inf
        for child in graph[state]:
            value = max(value, _minimax(state=child, graph=graph, reward=reward, depth=depth-1, maxplayer= not maxplayer))
        return value
    else:
        value = math.inf
        for child in graph[state]:
            value = min(value, _minimax(state=child, graph=graph, reward=reward, depth=depth - 1, maxplayer=not maxplayer))
        return value


class Queue(list):
    def __init__(self, *args, key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._key = key
        self.sort(key=self._key)
        self._set = set(self)

    def append(self, element):
        super().append(element)
        self.sort(key=self._key)
        self._set.add(element)

    def __contains__(self, item):
        if item in self._set:
            return True
        return False

    def pop(self, index=0):
        item = super().pop(index)
        self._set.remove(item)
        return item
