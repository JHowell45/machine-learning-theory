from dataclasses import dataclass
from typing import List, Union
from numpy import arange
from math import inf
from .cost_functions import squared_error
from tqdm import tqdm


@dataclass
class UnivariateLinearRegressionModel:
    gradient: float
    y_axis_shift: float = 0
    x_axis_shift: float = 0
    squared_error_score: float = None

    def predict(self, feature: float):
        return (self.gradient * (feature - self.x_axis_shift)) + self.y_axis_shift

    @classmethod
    def from_dataset(
        cls, features: List[Union[int, float]], labels: List[Union[int, float]]
    ):
        best_model = None
        best_squared_error_score = inf
        for test_gradient in tqdm(arange(0.1, 5, 0.1), desc="building model"):
            for y_shift in arange(-10, 10, 0.2):
                for x_shift in arange(-10, 10, 0.2):
                    model = cls(
                        gradient=test_gradient,
                        y_axis_shift=y_shift,
                        x_axis_shift=x_shift,
                    )
                    predicted_labels = [model.predict(feature) for feature in features]
                    model.squared_error_score = squared_error(labels, predicted_labels)
                    if model.squared_error_score < best_squared_error_score:
                        best_squared_error_score = model.squared_error_score
                        best_model = model
        return best_model
