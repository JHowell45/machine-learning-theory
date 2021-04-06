from dataclasses import dataclass, field
from numpy import array


@dataclass
class UnivariateLinearRegressionModel:
    gradient: float
    y_axis_shift: float = 0

    def predict(self, feature: float) -> float:
        return (self.gradient * feature) + self.y_axis_shift


@dataclass
class MultivariateLinearRegression:
    gradients: array = field(default_factory=array)
    y_axis_shift: float = 0

    def predict(self, features: array) -> float:
        return sum(self.gradients * features) + self.y_axis_shift
