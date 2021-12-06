"""
Algorithms for performing numerical optimization of (typically) convex functions
by estimating descent directions and updating parameters

All inputs to the algorithms are expected to be/accept NumPy arrays.

Algorithms receive some function as a parameter i.e. g, h. These functions are expected to
correspond to "mathematical" functions and take as input a single numpy array, of any dimension. This
argument should correspond to the parameter that you are seeking to optimize i.e. Theta, etc.

Plan code accordingly. The factory pattern is very useful for creating functions that have the correct signature at
runtime.
"""
import numpy as np


def gradient_descent(g, initial_point, alpha, iterations):
    """
    Performs gradient descent on a function. Takes the gradient of that function
    at a given parameter point and updates that parameter by some small fraction of
    the gradient value.

        X_1 = X_0 - alpha*g(X_0)

    where alpha is some small value.
    :param g: a Function(array) defining the gradient of the optimization target with respect to the optimization parameter
    :param initial_point: an Array representing the start value for the optimization parameter
    :param alpha: some small value, typically 0.01 is a good starting point but for clarity, the user is expected to
    provide a value. For ease of use, you might wrap this function with some default argument for alpha
    :param iterations: the number of iterations of gradient descent to perform
    :return: X, an Array of the final parameter values obtained
    :return: history, a List[Array] of all values of x. Useful for plotting the algorithms performance in the parameter
    space
    """
    history = [initial_point]
    X = initial_point

    for i in range(iterations):
        g_i = g(X)
        X_prime = X - alpha*g_i
        history.append(X_prime)
        X = X_prime

    return X, history


def newton_descent(g, h, initial_point, alpha, iterations):
    """
    Performs newtonian descent on a function. Takes the gradient of that function at a given point, normalizes
    the gradient by the hessian of the function at that point, and then updates the parameter:

        X_1 = X_0 - alpha * h(x_0)^-1 * g(X_0)

    where alpha is typically 1.

    :param g: a Function(array) defining the gradient of the optimization target with respect to the optimization parameter
    :param h: a Function(array) defining the hessian of the optimization target with respect to the optimization parameter
    :param initial_point: an Array representing the start value for the optimization parameter
    :param alpha: typically 1 for newtonian descent.
    :param iterations: the number of iterations of gradient descent to perform
    :return: X, an Array of the final parameter values obtained
    :return: history, a List[Array] of all values of x. Useful for plotting the algorithms performance in the parameter
    space
    """
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



