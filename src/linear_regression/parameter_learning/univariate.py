"""

"""
from math import inf

from pandas import Series
from tqdm import tqdm

from src.linear_regression.cost_functions import mean_squared_error
from src.linear_regression.models import UnivariateLinearRegressionModel


def batch_gradient_descent(
    features: Series,
    labels: Series,
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
            ) = single_gradient_descent(
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
                ) = single_gradient_descent(
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


def single_gradient_descent(
    features: Series,
    labels: Series,
    current_theta_0: float,
    current_theta_1: float,
    learning_rate: float,
):
    m = len(features)
    model = UnivariateLinearRegressionModel(current_theta_0, current_theta_1)
    # predicted_labels = model.multiple_predictions(features)  # somehow slower??
    predicted_labels = Series(model.predict(feature) for feature in features)

    theta_0_derivative = theta_0_partial_derivative(
        predictions=predicted_labels, actual_labels=labels, m=m
    )
    theta_1_derivative = theta_1_partial_derivative(
        predictions=predicted_labels, actual_labels=labels, features=features, m=m
    )

    current_theta_0 -= learning_rate * theta_0_derivative
    current_theta_1 -= learning_rate * theta_1_derivative
    cost_function_score = mean_squared_error(labels, predicted_labels)
    return current_theta_0, current_theta_1, cost_function_score


def theta_0_partial_derivative(
    predictions: Series, actual_labels: Series, m: int
) -> float:
    return (1 / m) * sum(predictions.subtract(actual_labels))


def theta_1_partial_derivative(
    predictions: Series, actual_labels: Series, features: Series, m: int
) -> float:
    return (1 / m) * sum(predictions.subtract(actual_labels).multiply(features))
