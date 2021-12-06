# TODO fix gd_momentum
from old.descent_methods.gradient_descent import GradientDescentOptimizer


class GDMomentum(GradientDescentOptimizer):
    def __init__(self, *args, gamma=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self._ps = []

    def p(self):
        """
        Returns momentum descent direction

        P_x1 = gamma*P_x0 + alpha*G(x1)

        momentum pushes most recent gradient direction in the direction of the
        previous gradient direction.
        :return:
        :rtype:
        """
        if len(self._ps) == 0:
            P_i = super().p()
        else:
            prev_p = self._ps[-1]
            X_i = self.parameters[-1]
            G_i = self.g(X_i)
            P_i = -self.alpha*G_i + self.gamma*prev_p

        self._ps.append(P_i)
        return P_i


if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)
    from old.descent_methods._functions import F4 as F
    from old.descent_methods._utils import surface_plot, fvalue_3D, param_plot
    f, g = F.f, F.g

    import matplotlib.pyplot as plt

    optim = GradientDescentOptimizer(f, g, alpha=0.1, initial_point=(2,2))
    for i in range(50):
        optim.step()

    optimM = GDMomentum(f, g, alpha=0.1, initial_point=(2,2))
    for i in range(50):
        optimM.step()


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    p1 = fvalue_3D(f, optim.parameters, ax=ax, label="GD")
    p2 = fvalue_3D(f, optimM.parameters, ax=ax, label="Momentum")

    XSTART = -2
    XEND = 2
    YSTART = -2
    YEND = 2

    surf = surface_plot(f, ax=ax, xstart=-XSTART, xend=XEND, ystart=-YSTART, yend=YEND, alpha=0.2)
    plt.title('Gradient Descent w/ Momentum')
    plt.legend(handles=[p1, p2])
    plt.show()

    fig, ax = plt.subplots()
    param_plot(f, optim.parameters, ax=ax, xstart=-2, ystart=-2, xend=2, yend=2)
    param_plot(f, optimM.parameters, ax=ax, contour=False, xstart=-2, ystart=-2, xend=2, yend=2)
    plt.show()
    for param1, param2 in zip(optim.parameters, optimM.parameters):
        print(param1, param2)


