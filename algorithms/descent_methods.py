"""
Algorithms for performing numerical optimization of (typically) convex functions
by estimating descent directions and updating parameters

All inputs to the algorithms are expected to be/accept NumPy arrays. If I have time I will
implement proper argument handling

"""
import numpy as np


def gradient_descent(g, initial_point, alpha, iterations):
    history = [initial_point]
    X = initial_point

    for i in range(iterations):
        g_i = g(X)
        X_prime = X - alpha*g_i
        history.append(X_prime)
        X = X_prime

    return X, history


def newton_descent(g, h, initial_point, alpha, iterations):
    history = [initial_point]
    X = initial_point

    for i in range(iterations):
        p_i = np.matmul(np.linalg.inv(h(X)), g(X))
        X_prime = X - alpha*p_i
        history.append(X_prime)
        X = X_prime

    return X, history


def conjugate_gradient(g, initial_point, alpha, iterations):
    # TODO
    pass



