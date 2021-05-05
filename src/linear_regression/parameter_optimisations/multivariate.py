"""

"""
from math import inf

from pandas import DataFrame, Series
from tqdm import tqdm

from linear_regression.cost_functions import mean_squared_error
from linear_regression.models import MultivariateLinearRegression

from .univariate import theta_0_partial_derivative


def batch_gradient_descent(
    features: DataFrame,
    labels: DataFrame,
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
