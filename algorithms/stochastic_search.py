"""
Algorithms for performing optimization of black box functions
using sampling-based methods to iteratively find better solutions

All inputs to the algorithms are expected to be/accept NumPy arrays. If I have time I will
implement proper argument handling
"""
import numpy as np
import random


def simulated_annealing(f, initial_point, schedule, iterations):
    """
     Performs simulated annealing on a function to search for a global minimum,
    given an initial point The function signature of f is expected to take as positional arguments
    the same number of elements as contained in initial point
    :param f: The function to be optimized
    :param initial_point: an initial point - np.array is expected
    :param schedule: The cooling schedule, a function that accepts the iteration and returns the temperature parameter
    :param iterations: The number of iterations to run the simulation for
    :param seed: random seed
    :return:
    """

    history = [initial_point]
    candidates = [initial_point]
    X = initial_point
    Y = f(X)

    for i in range(iterations):

        X_cand = np.random.multivariate_normal(X, np.identity(len(X)))
        Y_cand = f(X_cand)

        if Y_cand >= Y:
            t = schedule(i)
            prob = np.exp((Y - Y_cand)/t)
            accept = random.choices((True, False), (prob, 1-prob))[0]
        else:
            accept = True

        candidates.append(X_cand)
        if accept:
            history.append(X_cand)
            X = X_cand
            Y = Y_cand

    return X, history, candidates


def crossentropy(f, initial_dist, k, threshold, iterations):
    """
    Performs crossentropy search
    :param f:
    :param initial_point:
    :param iterations:
    :param seed:
    :return:
    """

    history = [initial_dist]
    populations = []
    elites = []

    mu, sigma = initial_dist
    normal = MultivariateNormal(mu, sigma)

    for i in range(iterations):

        samples = normal.sample(k)
        elite = np.array(sorted(samples, key=lambda x: f(x))[:int(threshold*k)])
        populations.append(samples)
        elites.append(elite)

        m = MultivariateNormal()
        m.fit(elite)
        mu, sigma = m.u.squeeze(), m.sig

        history.append((mu, sigma))

    return (mu, sigma), history, populations, elites


def search_gradient(f, initial_dist, k, iterations, alpha):
    """
    Performs search gradient on a function of arbitrary number of parameters
    Search gradient works by updating a parametrized distribution (gaussian in
    this case) via gradient descent. It computes the gradient of the probability
    density function with respect to a mean mu and a covariance sigma, and updates
    by the negative gradient * some hyperparameter alpha
    :param f:
    :type f:
    :param initial_dist:
    :type initial_dist:
    :param steps:
    :type steps:
    :return:
    :rtype:
    """

    history = [initial_dist]
    populations = []

    mu, sigma = initial_dist

    for i in range(iterations):
        normal = MultivariateNormal(mu, sigma, f)
        samples = normal.sample(k)
        dmu = sum([normal.dmu(sample) for sample in samples])/k
        dsigma = sum([normal.dsigma(sample) for sample in samples]) / k

        # normalizing the gradient for better performance
        dmu = dmu / np.linalg.norm(dmu)
        dsigma = dsigma/np.linalg.norm(dsigma)

        mu = mu-alpha*dmu
        sigma = sigma - alpha*dsigma

        history.append((mu,sigma))
        populations.append(samples)

    return (mu, sigma), history, populations


class MultivariateNormal:
    def __init__(self, u=None, sig=None, f=None):
        if u is not None and sig is not None:
            self.u = np.array(u).reshape((2,1))
            self.sig = np.array(sig)
        else:
            self.u = None
            self.sig = None

        self.f = f
        if f is not None:
            self._set_derivatives()

    def fit(self, x):
        x = x[..., np.newaxis] if x.ndim == 2 else x
        self.u = x.mean(0)
        self.sig = np.einsum('ijk,ikj->jk', x - self.u, x - self.u) / (x.shape[0] - 1)
        self._set_derivatives()

    def prob(self, x):
        x = x[..., np.newaxis] if x.ndim == 2 else x
        left = (2 * np.pi) ** (-self.u.shape[0] / 2) * np.linalg.det(self.sig) ** (-1 / 2)
        right = np.exp((-1 / 2) * np.einsum('ijk,jl,ilk->ik', x - self.u, np.linalg.inv(self.sig), x - self.u))
        return left * right

    def sample(self, k):
        u = self.u.squeeze()
        return np.array([np.random.multivariate_normal(u, self.sig) for _ in range(k)])

    def _ddistdmu_factory(self):
        mu = self.u
        sigma = self.sig
        def inner(x):
            right = self.f(x)
            left = np.array(x) - mu.squeeze()
            left = np.matmul(np.linalg.inv(sigma), left)
            return left * right

        return inner

    def _ddistdsigma_factory(self):
        mu = self.u
        sigma = self.sig
        def inner(x):
            right = self.f(x)
            left = 0.5 * np.linalg.inv(sigma)
            diff = x - mu.squeeze()
            left = np.matmul(left, diff)
            left = np.outer(left, left)
            left = np.matmul(left, np.linalg.inv(sigma))
            return left * right

        return inner

    def _set_derivatives(self):
        self.dmu = self._ddistdmu_factory()
        self.dsigma = self._ddistdsigma_factory()

    def dmu(self, *args):
        raise NotImplementedError("Has not been defined for this instance")

    def dsigma(self, *args):
        raise NotImplementedError("Has not been defined for this instance")