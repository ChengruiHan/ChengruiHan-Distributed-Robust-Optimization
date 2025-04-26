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


class G1(Function):
    def value(self, x):
        return x - 0.6  # g1(x) = x - 0.6 <= 0

    def subgradient(self, x):
        return 1  # The gradient is constant 1


class G2(Function):
    def value(self, x):
        return x ** 2 - 0.3  # g2(x) = x^2 - 0.3 <= 0

    def subgradient(self, x):
        return 2 * x  # The gradient is 2x


class G3(Function):
    def value(self, x):
        import numpy as np
        return np.exp(x) - 1.4  # g3(x) = e^x - 1.4 <= 0

    def subgradient(self, x):
        import numpy as np
        return np.exp(x)  # The gradient of e^x is e^x


class G4(Function):
    def value(self, x):
        return (x - 0.2) ** 2 - 0.01  # g4(x) = (x-0.2)^2 - 0.01 <= 0

    def subgradient(self, x):
        return 2 * (x - 0.2)  # The gradient is 2(x-0.2)