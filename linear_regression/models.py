from dataclasses import dataclass


@dataclass
class UnivariateLinearRegressionModel:
    gradient: float
    y_axis_shift: float = 0

    def predict(self, feature: float) -> float:
        return (self.gradient * feature) + self.y_axis_shift
