"""

"""
from math import inf
from typing import List, Union

from tqdm import tqdm
from .cost_functions import mean_squared_error
from .models import UnivariateLinearRegressionModel
from pandas import Series


def ulr_batch_gradient_descent(
    features: Union[Series, List[Union[float, int]]],
    labels: Union[Series, List[Union[float, int]]],
    current_theta_0: int = 0,
    current_theta_1: int = 0,
    learning_rate: float = 0.0001,
    epochs: int = None,
):
    previous_mse_score = inf
    current_mse_score = 0
    rounds = 0
    start = True
    if epochs is None:
        while previous_mse_score > current_mse_score:
            if start:
                current_mse_score = inf
                start = False
            previous_mse_score = current_mse_score
            (
                current_theta_0,
                current_theta_1,
                current_mse_score,
            ) = _single_gradient_descent(
                features, labels, current_theta_0, current_theta_1, learning_rate
            )
            rounds += 1
            print(
                {
                    "current_theta_0": round(current_theta_0, 2),
                    "current_theta_1": round(current_theta_1, 2),
                    "current_mse_score": round(current_mse_score, 4),
                    "epochs": rounds,
                }
            )
    else:
        for _ in tqdm(range(epochs)):
            if previous_mse_score > current_mse_score:
                if start:
                    current_mse_score = inf
                    start = False
                previous_mse_score = current_mse_score

                (
                    current_theta_0,
                    current_theta_1,
                    current_mse_score,
                ) = _single_gradient_descent(
                    features, labels, current_theta_0, current_theta_1, learning_rate
                )
                rounds += 1
            else:
                break
    return {
        "current_theta_0": round(current_theta_0, 2),
        "current_theta_1": round(current_theta_1, 2),
        "current_mse_score": round(current_mse_score, 4),
        "epochs": rounds,
    }


def _single_gradient_descent(
    features: List[Union[float, int]],
    labels: List[Union[float, int]],
    current_theta_0: float,
    current_theta_1: float,
    learning_rate: float,
):
    m = len(features)
    model = UnivariateLinearRegressionModel(current_theta_0, current_theta_1)
    # predicted_labels = model.multiple_predictions(features)  # somehow slower??
    predicted_labels = [model.predict(feature) for feature in features]

    theta_0_derivative = (
        sum((predicted - actual) for predicted, actual in zip(predicted_labels, labels))
        / m
    )
    theta_1_derivative = (
        sum(
            (predicted - actual) * feature
            for predicted, actual, feature in zip(predicted_labels, labels, features)
        )
        / m
    )

    current_theta_0 -= learning_rate * theta_0_derivative
    current_theta_1 -= learning_rate * theta_1_derivative
    cost_function_score = mean_squared_error(labels, predicted_labels)
    return current_theta_0, current_theta_1, cost_function_score
