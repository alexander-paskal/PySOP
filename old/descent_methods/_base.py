"""
Abstract base class for descent optimizers
"""
import abc
import typing as tp
from inspect import signature, _empty
import numpy as np


class BaseDescentOptimizer(abc.ABC):

    def __init__(self, f):
        self.validate_input_f(f)

    @abc.abstractmethod
    def step(self):
        pass

    @property
    @abc.abstractmethod
    def parameters(self) -> tp.List:
        raise NotImplementedError

    @staticmethod
    def infer_dims(f):
        """
        returns the number of inputs expected from a received
        callable
        :param f:
        :type f:
        :return:
        :rtype:
        """
        return len(signature(f).parameters.keys())

    @staticmethod
    def validate_input_f(f):
        """
        Validates a function to ensure it is a valid optimization target
        :return:
        :rtype:
        """
        # TODO implement function validation
        pass

    def random_point(self, dim):
        """
        Returns a random point of dimensionality dim
        :param dim:
        :type dim:
        :return:
        :rtype:
        """
        return np.random.random_sample(dim)