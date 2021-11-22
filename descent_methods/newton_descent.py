"""
this module contains utilities for performing newton descent on a function
"""
import numpy as np
from descent_methods._base import BaseDescentOptimizer


class NewtonDescentOptimizer(BaseDescentOptimizer):
    def __init__(self, f, g, h, initial_point=None, alpha=1):
        super().__init__(f)
        self.f = f
        self.g = g
        self.h = h

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

        X_i = self._parameters[-1]

        G_i = self.g(*X_i)
        H_i = self.h(*X_i)
        try:
            Hinv_i = np.linalg.inv(H_i)
        except np.linalg.LinAlgError:  # singular hessian
            # TODO handle this error
            print("Singular Hessian Error")
            return X_i

        P_i = np.matmul(-1 * (Hinv_i), G_i)  # normalize the gradient by the hessian
        P_i *= self.alpha
        X_iPlusOne = X_i + P_i
        self._parameters.append(X_iPlusOne)
        return X_iPlusOne

    @property
    def parameters(self):
        return self._parameters


if __name__ == '__main__':
    from descent_methods._functions import F1, F2
    from descent_methods._utils import param_plot
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1,2)

    for i, F in enumerate((F1, F2)):
        optim = NewtonDescentOptimizer(F.f, F.g, F.h, initial_point=(2, 2))
        for _ in range(10):
            optim.step()
        x, y = zip(*optim.parameters)
        param_plot(F.f, optim.parameters, ax=axs[i])
        print((optim.parameters))

    plt.suptitle("Newton's Descent")
    axs[0].set_title("x^2 + y^2")
    axs[1].set_title("x^4 + 3y^4")
    plt.show()