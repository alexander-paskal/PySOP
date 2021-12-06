import numpy as np


class BaseFunction:
    @staticmethod
    def f(x1, x2):
        raise NotImplementedError

    @staticmethod
    def g(x1, x2):
        raise NotImplementedError

    @staticmethod
    def h(x1, x2):
        raise NotImplementedError


# f(x, y ) = x^2 + y^2
class F1(BaseFunction):
    @staticmethod
    def f(x1, x2):
        return x1**2 + x2**2

    @staticmethod
    def g(x1, x2):
        return [
            2*x1, 2*x2
        ]

    @staticmethod
    def h(x1, x2):
        return [
            [2, 0],
            [0, 2]
        ]


# f(x, y) = x^4 + y^4
class F2(BaseFunction):
    @staticmethod
    def f(x1, x2):
        return x1**4 + 3*x2**4

    @staticmethod
    def g(x1, x2):
        return [
            4*x1**3, 12*x2**3
        ]

    @staticmethod
    def h(x1, x2):
        return [
            [12*x1**2, 0],
            [0, 36*x2**2]
        ]


# drop wave function
class F3(BaseFunction):
    @staticmethod
    def f(x1, x2):
        top = 1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2))
        bottom = 0.5 * (x1 ** 2 + x2 ** 2) + 2
        return -top / bottom

    @staticmethod
    def g(x1, x2):
        return np.array([F3.df2dx1(x1, x2), F3.df2dx2(x1, x2)])

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


# f(x, y ) = x^2 + 4y^2
class F4(BaseFunction):
    @staticmethod
    def f(x1, x2):
        return x1**2 + 4*x2**2

    @staticmethod
    def g(x1, x2):
        return [
            2*x1, 8*x2
        ]

    @staticmethod
    def h(x1, x2):
        return [
            [2, 0],
            [0, 8]
        ]


# f(x, y ) = x^2 + 10y^2
class F5(BaseFunction):
    @staticmethod
    def f(x1, x2):
        return x1**2 + 10*x2**2

    @staticmethod
    def g(x1, x2):
        return [
            2*x1, 20*x2
        ]

    @staticmethod
    def h(x1, x2):
        return [
            [2, 0],
            [0, 20]
        ]