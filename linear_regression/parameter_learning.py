"""

"""
from .models import UnivariateLinearRegressionModel
from .cost_functions import mean_squared_error
from numpy import arange
from math import inf
from typing import List, Union
from tqdm import tqdm


def ulr_batch_gradient_descent(
    features: List[Union[float, int]],
    labels: List[Union[float, int]],
    current_gradient: int = 0,
    current_y_shift: int = 0,
    learning_rate: float = 0.0001,
    epochs: int = None,
):
    previous_score = inf
    current_score = inf
    rounds = 0
    if epochs is None:
        while previous_score <= current_score:
            previous_score = current_score
            current_gradient, current_y_shift, current_score = _single_gradient_descent(
                features, labels, current_gradient, current_y_shift, learning_rate
            )
            rounds += 1
    else:
        for _ in tqdm(range(epochs)):
            current_gradient, current_y_shift, current_score = _single_gradient_descent(
                features, labels, current_gradient, current_y_shift, learning_rate
            )
            rounds += 1
    return current_gradient, current_y_shift, current_score, rounds


def _single_gradient_descent(
    features: List[Union[float, int]],
    labels: List[Union[float, int]],
    gradient: float,
    y_shift: float,
    learning_rate: float,
):
    m = len(features)
    model = UnivariateLinearRegressionModel(gradient, y_shift)
    predicted_labels = [model.predict(feature) for feature in features]
    gradiant_derivative = (
        sum((predicted - actual) for predicted, actual in zip(predicted_labels, labels))
        / m
    )
    y_axis_derivative = (
        sum(
            (predicted - actual) * feature
            for predicted, actual, feature in zip(predicted_labels, labels, features)
        )
        / m
    )
    gradient -= learning_rate * gradiant_derivative
    y_shift -= learning_rate * y_axis_derivative
    cost_function_score = mean_squared_error(labels, predicted_labels)
    return gradient, y_shift, cost_function_score
