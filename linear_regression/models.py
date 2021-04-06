from dataclasses import dataclass, field
from pandas import Series


@dataclass
class UnivariateLinearRegressionModel:
    gradient: float
    y_axis_shift: float = 0

    def predict(self, feature: float) -> float:
        return (self.gradient * feature) + self.y_axis_shift


@dataclass
class MultivariateLinearRegression:
    gradients: Series = field(default_factory=Series)
    y_axis_shift: float = 0

    def predict(self, features: Series) -> float:
        return self.gradients.dot(features) + self.y_axis_shift
