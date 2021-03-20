from dataclasses import dataclass


@dataclass
class UnivariateLinearRegressionModel:
    gradient: float
    y_axis_shift: float = 0
    x_axis_shift: float = 0

    def predict(self, feature: float):
        return (self.gradient * (feature - self.x_axis_shift)) + self.y_axis_shift
