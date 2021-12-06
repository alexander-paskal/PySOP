"""
Utility functions used in descent methods
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import seaborn as sns

sns.set_theme()


def show_plot(*oargs, **okwargs):
    """
    Decorator for eagerly showing plots
    :param args:
    :type args:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    def decorator(fn):
        def inner(*iargs, ax=None, **ikwargs):
            fig = None
            if ax is None:
                fig, ax = plt.subplots(*oargs, **okwargs)
            result = fn(*iargs, ax=ax, **ikwargs)
            if fig is not None:
                plt.show()
            return result
        return inner
    return decorator




@show_plot
def function_plot(f, parameters, ax):
    """
    Creates a plot of the function parameters
    :param f:
    :type f:
    :param parameters:
    :type parameters:
    :param ax:
    :type ax:
    :return:
    :rtype:
    """

    fvalues = [f(*param) for param in parameters]
    result = ax.scatter()
    return result


@show_plot(subplot_kw={"projection": "3d"})
def surface_plot(f, ax, xstart=-5, xend=5, ystart=-5, yend=5, res=101, cmap = cm.get_cmap("GnBu"), **kwargs):
    """
    Assumes ax was created with proper keyword
    :return:
    :rtype:
    """
    if "cmap" in kwargs:
        cmap = kwargs.pop("cmap")

    x = np.linspace(xstart, xend, res)
    y = np.linspace(ystart, yend, res)
    X, Y = np.meshgrid(x, y)

    Z = f(X, Y)

    result = ax.plot_surface(X, Y, Z, linewidth=0, cmap= cmap, antialiased=False, **kwargs)  # cmap = cm.coolwarm

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    return result
    # fig.colorbar(surf, shrink=0.5, aspect=5)


@show_plot()
def param_plot(f, params, ax, contour=True, xstart=-5, xend=5, ystart=-5, yend=5, res=101, **kwargs):
    """
    Creates a parameter plot for a given function.
    Displays the parameters on top of the function contours
    :param params:
    :type params:
    :param ax:
    :type ax:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """

    if contour:
        x = np.linspace(xstart, xend, res)
        y = np.linspace(ystart, yend, res)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        ax.contour(X, Y, Z)
    X, Y = zip(*params)
    ax.plot(X, Y, "-o", **kwargs)



@show_plot(subplot_kw={"projection": "3d"})
def fvalue_3D(f, parameters, ax, **kwargs):

    X, Y = zip(*parameters)
    F = [f(x, y) for x, y in parameters]

    result = ax.plot(X, Y, F, '-o', **kwargs)
    return result[0]




if __name__ == '__main__':
    def f(x1, x2):
        return x1**2 #+ 2*x1*x2* + x2**2


    surface_plot(f)

