"""
Creates visualizations for different descent methods
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from algorithms.descent_methods import gradient_descent, newton_descent

sns.set_theme()


class F1:
    @staticmethod
    def f(x):
        return np.sum(x**2, axis=0)

    @staticmethod
    def g(x):
        return 2*x

    @staticmethod
    def h(x):
        return np.zeros(x.shape[0]) + 2 * np.identity(x.shape[0])


class F2:
    @staticmethod
    def f(x):
        return x[0]**2 + 10*x[1]**2

    @staticmethod
    def g(x):
        return np.array([2*x[0], 20*x[1]])

    @staticmethod
    def h(x):
        return np.array([
            [2, 0],
            [0, 20]
        ])


class F3:
    @staticmethod
    def f(x):
        return x[0]**4 + 2*x[1]**4

    @staticmethod
    def g(x):
        return np.array([
            4*x[0]**3, 8*x[1]**3
        ])

    @staticmethod
    def h(x):
        return np.array([
            [12*x[0]**2, 0],
            [0, 24*x[1]**2]
        ])


def plt_surf(f, **kwargs):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x = np.linspace(-10, 10, 101)
    y = np.linspace(-10, 10, 101)
    X, Y = np.meshgrid(x, y)

    Z = f(np.array([X, Y]))

    result = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, **kwargs)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    return result


def plt_line3d(f, params, *args, **kwargs):

    F = [f(param) for param in params]
    X, Y = zip(*params)

    result = plt.plot(X, Y, F, *args, **kwargs)
    return result


def gd_bowl():
    """
    Visualizes gradient descent on a perfectly round, convex function
    :return:
    """
    plt_surf(F1.f, alpha=0.3, cmap="Greens")
    ax = plt.gca()
    ax.view_init(azim=-44, elev=35)
    IP = np.array((10, 10))
    ALPHA = 0.1
    ITERATIONS = 10

    params, history = gradient_descent(F1.g, initial_point=IP, alpha=ALPHA, iterations=ITERATIONS)
    plt_line3d(F1.f, history, "-o")
    plt.title("Gradient Descent")
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("F(x,y)", rotation=90)
    ax.set_zticks([])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.savefig("images/gradient_descent_bowl.png")
    plt.show()


def gd_ellipse():
    """
    Visualizes gradient descent on a more ellipsoidic function,
    and the impact that alpha can have on convergence
    :return:
    """

    alphas = (0.1, 0.01, 0.001, 0.0001)

    for alpha in alphas:
        plt_surf(F2.f, alpha=0.3, cmap="Greens")
        ax = plt.gca()
        ax.view_init(azim=-44, elev=35)
        IP = np.array((10, 10))
        ITERATIONS = 30

        params, history = gradient_descent(F2.g, initial_point=IP, alpha=alpha, iterations=ITERATIONS)
        plt_line3d(F2.f, history, "-o")
        plt.title(f"Gradient Descent: alpha = {alpha}")
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("F(x,y)", rotation=90)
        ax.set_zticks([])
        ax.set_yticks([])
        ax.set_xticks([])
        plt.savefig(f"images/gradient_descent_ellipse_alpha={alpha}.png")
        plt.show()


def nd_ellipse():
    plt_surf(F2.f, alpha=0.3, cmap="Greens")
    ax = plt.gca()
    ax.view_init(azim=-44, elev=35)
    IP = np.array((10, 10))
    ALPHA = 1
    ITERATIONS = 30

    params, history = newton_descent(F2.g, F2.h, initial_point=IP, alpha=ALPHA, iterations=ITERATIONS)
    plt_line3d(F2.f, history, "-o")
    plt.title(f"Newton Descent")
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("F(x,y)", rotation=90)
    ax.set_zticks([])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.savefig(f"images/newton_descent_ellipse.png")
    plt.show()


def higher_order():
    # newton descent
    plt_surf(F3.f, alpha=0.3, cmap="Greens")
    ax = plt.gca()
    ax.view_init(azim=-44, elev=35)
    IP = np.array((10, 10))
    ALPHA = 1
    ITERATIONS = 10

    params, history = newton_descent(F3.g, F3.h, initial_point=IP, alpha=ALPHA, iterations=ITERATIONS)
    plt_line3d(F3.f, history, "-o")
    plt.title(f"Newton Descent - Higher Order Function")
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("F(x,y)", rotation=90)
    ax.set_zticks([])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.savefig(f"images/newton_descent_higher_order.png")
    plt.show()

    # gradient descent
    plt_surf(F3.f, alpha=0.3, cmap="Greens")
    ax = plt.gca()
    ax.view_init(azim=-44, elev=35)
    IP = np.array((10, 10))
    ALPHA = 0.0005
    ITERATIONS = 10

    params, history = gradient_descent(F3.g, initial_point=IP, alpha=ALPHA, iterations=ITERATIONS)
    plt_line3d(F3.f, history, "-o")
    plt.title(f"Gradient Descent - Higher Order Function, alpha = 0.0005")
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("F(x,y)", rotation=90)
    ax.set_zticks([])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.savefig(f"images/gradient_descent_higher_order.png")
    plt.show()


if __name__ == '__main__':
    gd_bowl()
    gd_ellipse()
    nd_ellipse()
    higher_order()



