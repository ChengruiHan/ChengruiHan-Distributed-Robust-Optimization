import numpy as np


class Function:
    """
    Base class: Represents a function and its subgradient.
    """
    def value(self, x):
        """
        Compute the function value.
        """
        raise NotImplementedError("Subclasses must implement 'value' method.")

    def subgradient(self, x):
        """
        Compute the subgradient.
        """
        raise NotImplementedError("Subclasses must implement 'subgradient' method.")


class F1(Function):

    def value(self, x):
        return -4*x + 2*x**2

    def subgradient(self, x):
        return -4 + 4*x


class F2(Function):

    def value(self, x):
        return x**2

    def subgradient(self, x):
        return 2 * x  # The gradient is 2x


class F3(Function):

    def value(self, x):
        return 3*x**2

    def subgradient(self, x):
        return 6*x


class F4(Function):

    def value(self, x):
        return 2*x**2

    def subgradient(self, x):
        return 4*x

