"""
Creates visualizations for stochastic search methods
"""

import numpy as np
from algorithms.stochastic_search import search_gradient, crossentropy, simulated_annealing, MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator




class DropWave:
    @staticmethod
    def f(x):
        x1, x2 = x[0], x[1]
        top = 1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2))
        bottom = 0.5 * (x1 ** 2 + x2 ** 2) + 2
        return -top / bottom

    @staticmethod
    def g(x):
        x1, x2 = x[0], x[1]
        return np.array([DropWave.df2dx1(x1, x2), DropWave.df2dx2(x1, x2)])

    @staticmethod
    def df2dx1(x1, x2):
        top = -(1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2)))
        bottom = 0.5 * (x1 ** 2 + x2 ** 2) + 2

        dtopH = np.sin(12 * np.sqrt(x1 ** 2 + x2 ** 2))
        dtopG = 6 / np.sqrt(x1 ** 2 + x2 ** 2)
        dtopF = 2 * x1

        dtop = dtopH * dtopG * dtopF

        dbottom = x1

        return (dtop * bottom - dbottom * top) / bottom ** 2

    @staticmethod
    def df2dx2(x1, x2):
        top = -(1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2)))
        bottom = 0.5 * (x1 ** 2 + x2 ** 2) + 2

        dtopH = np.sin(12 * np.sqrt(x1 ** 2 + x2 ** 2))
        dtopG = 6 / np.sqrt(x1 ** 2 + x2 ** 2)
        dtopF = 2 * x2

        dtop = dtopH * dtopG * dtopF

        dbottom = x2

        return (dtop * bottom - dbottom * top) / bottom ** 2



def plt_surf(f, **kwargs):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x = np.linspace(-10, 10, 101)
    y = np.linspace(-10, 10, 101)
    X, Y = np.meshgrid(x, y)

    Z = f(np.array([X, Y]))

    result = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, **kwargs)

    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("F(x,y)", rotation=90)
    ax.set_zticks([])
    ax.set_yticks([])
    ax.set_xticks([])
    return result


def plt_line3d(f, params, *args, **kwargs):

    F = [f(param) for param in params]
    X, Y = zip(*params)

    result = plt.plot(X, Y, F, *args, **kwargs)
    return result


def plot_dropwave():
    plt_surf(DropWave.f, cmap="Reds", alpha=0.5)
    plt.savefig("images/dropwave.png")
    plt.title("DropWave Function")
    plt.show()


def simulated_annealing_dw():
    """
    Performs and visualizes simulated annealing on the drop-wave function
    :return:
    """

    IP = np.array([2,2])
    T = 5
    def schedule(i):
        return T / (i+1)
    ITERATIONS = 100
    ATTEMPTS = 5

    plt_surf(DropWave.f, cmap="Reds", alpha=0.04)
    ax = plt.gca()
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.1f}')
    winner = np.array([-5, -5])
    for i in range(ATTEMPTS):
        best, history, candidates = simulated_annealing(DropWave.f, initial_point=IP,schedule=schedule, iterations=ITERATIONS)
        plt_line3d(DropWave.f, history, "-o", markersize=3)
        if DropWave.f(best) < DropWave.f(winner):
            winner = best
    plt.title("Simulated Annealing\n"
              "Schedule = Fast Cooling\n"
              )
    plt.savefig("images/simulated_annealing_dropwave.png")
    plt.show()


def crossentropy_dropwave(k, threshold):
    K = k
    INITIAL_DIST = np.array([-5,-5]), np.identity(2)
    THRESHOLD = threshold
    ITERATIONS = 100
    ATTEMPTS = 5

    winner = np.array([-10, -10])
    plt_surf(DropWave.f, cmap="Reds", alpha=0.04)
    ax = plt.gca()
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.1f}')
    for i in range(ATTEMPTS):
        dist, history, populations, elites = crossentropy(DropWave.f, INITIAL_DIST, K, THRESHOLD, ITERATIONS)
        mu, sigma = dist

        plt_line3d(DropWave.f, list(zip(*history))[0], "-o", markersize=3)

        if DropWave.f(mu) < DropWave.f(winner):
            winner = mu
    plt.title(f"Cross Entropy\nInitial Mean = {INITIAL_DIST[0].tolist()}")
    plt.savefig(f"images/crossentropy_dropwave_{INITIAL_DIST[0].tolist()}.png")
    plt.show()


def search_gradient_dw():
    K = 50
    INITIAL_DIST = np.array([-2,-2]), np.identity(2)
    ITERATIONS = 100
    ATTEMPTS = 5
    ALPHA = 0.1

    winner = np.array([-10, -10])
    winner_val = DropWave.f(winner)

    plt_surf(DropWave.f, cmap="Reds", alpha=0.04)
    ax = plt.gca()
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.1f}')
    for i in range(ATTEMPTS):
        dist, history, populations = search_gradient(DropWave.f, INITIAL_DIST, K, ITERATIONS, ALPHA)
        mu, sigma = dist

        plt_line3d(DropWave.f, list(zip(*history))[0], "-o", markersize=3)
        mu_val = DropWave.f(mu)
        if mu_val < winner_val:
            winner = mu
            winner_val = mu_val

    plt.title(f"Search Gradient\nAlpha={ALPHA}, Iterations={ITERATIONS}\n Initial Mean = {INITIAL_DIST[0].tolist()}")
    plt.savefig(f"images/search_gradient_dropwave_{INITIAL_DIST[0].tolist()}.png")
    plt.show()


if __name__ == '__main__':
    # plot_dropwave()
    # simulated_annealing_dw()
    # crossentropy_dropwave(100, 0.2)
    # crossentropy_dropwave(100, 0.7)
    # crossentropy_dropwave(20, 0.8)
    crossentropy_dropwave(20, 0.2)
    search_gradient_dw()