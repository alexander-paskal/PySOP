"""
This module contains utilities for performing gradient descent on a function
"""
import numpy as np
from descent_methods._base import BaseDescentOptimizer


class GradientDescentOptimizer(BaseDescentOptimizer):
    def __init__(self, f, g, initial_point=None, alpha=1):
        super().__init__(f)
        self.f = lambda x: np.array(f(*x))
        self.g = lambda x: np.array(g(*x))
        self.dim = self.infer_dims(f)
        self._parameters = []
        self._alpha = alpha

        if initial_point is None:
            initial_point = self.random_point(self.dim)
        self._parameters.append(np.array(initial_point))

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not 0 < value <= 1:
            raise ValueError("Invalid alpha value, please select an alpha on range (0, 1]")
        self._alpha = value

    def step(self, alpha=None):
        """
        reduces the last parameter by newton descent, calculates a new parameter
        :param alpha:
        :type alpha:
        :return:
        :rtype:
        """
        if alpha is not None:
            self.alpha = alpha

        X_i = self.parameters[-1]
        P_i = self.p()
        X_iPlusOne = X_i + P_i
        self.parameters.append(X_iPlusOne)
        return X_iPlusOne

    def p(self):
        """
        Returns the positional vector
        :param g_i:
        :type g_i:
        :return:
        :rtype:
        """
        X_i = self.parameters[-1]

        G_i = np.array(self.g(X_i))
        return -self.alpha * G_i

    @property
    def parameters(self):
        return self._parameters


if __name__ == '__main__':
    np.random.seed(1)
    from descent_methods._functions import F4 as F
    from descent_methods._utils import surface_plot, fvalue_3D, param_plot
    f, g = F.f, F.g

    import matplotlib.pyplot as plt
    optim = GradientDescentOptimizer(f, g, alpha=0.2, initial_point=(0.8, 0.8))
    for i in range(500):
        optim.step()


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surface_plot(f, ax=ax, xstart=-0.8, xend=0.8, ystart=-0.8, yend=0.8, alpha=0.2)
    fvalue_3D(f, optim.parameters, ax=ax)
    plt.title('Gradient Descent')
    plt.show()


    learning_rates = (0.25, 0.2, 0.1, 0.01)

    fig, axs = plt.subplots(2,2)
    for i, alpha in enumerate(learning_rates):
        ax = axs.flat[i]

        optim = GradientDescentOptimizer(f, g, alpha=alpha, initial_point=(0.8, 0.8))
        for i in range(500):
            optim.step()

        param_plot(f, optim.parameters, ax=ax, xstart=-1, ystart=-1, xend=1, yend=1)
        ax.set_title(f"alpha = {alpha}")
        if i < 2:
            ax.tick_params(labelbottom=False)

    plt.suptitle("Gradient Descent Learning Rates\nx^2 + 4y^2")
    plt.show()


