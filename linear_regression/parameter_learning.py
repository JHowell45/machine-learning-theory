"""

"""
from math import inf
from typing import List, Union

from tqdm import tqdm
from pandas import DataFrame
from numpy import array
from .cost_functions import mean_squared_error
from .models import UnivariateLinearRegressionModel, MultivariateLinearRegression


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
        while previous_score >= current_score:
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
    return {
        "current_gradient": round(current_gradient, 2),
        "current_y_shift": round(current_y_shift, 2),
        "current_score": round(current_score, 4),
        "epochs": rounds,
    }


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


def mlr_batch_gradient_descent(
    features: DataFrame, labels: array, current_gradients: array, learning_rate: float
):
    previous_score = inf
    current_score = inf
    rounds = 0
    while previous_score >= current_score:
        previous_score = current_score
        current_gradients, current_score = ()
        rounds += 1
    return {
        "current_gradients": array([round(x, 2) for x in current_gradients]),
        "current_score": round(current_score, 4),
        "epochs": rounds,
    }


def __single_mlr_gradient_descent(
    features: DataFrame, labels: array, gradients: array, learning_rate: float
):
    m = len(features)
    model = MultivariateLinearRegression(gradients=gradients)
    predicted_labels = []
    for gradient_index in range(0, m):
        predicted_label = model.predict(features.transpose().iloc[gradient_index])
        predicted_labels.append(predicted_label)

        values = []
        for predicted, actual, feature in zip(predicted_labels, labels.features):
            values.append((predicted - actual) * feature[1])
        gradiant_derivative = (
            sum(
                (predicted - actual)
                for predicted, actual, feature in zip(predicted_labels, labels.features)
            )
            / m
        )
