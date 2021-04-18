"""

"""
import pytest

from pandas import Series

from src.linear_regression.models import (
    MultivariateLinearRegression,
    UnivariateLinearRegressionModel,
)


class TestUnivariateLinearRegressionModel:
    @pytest.mark.parametrize(
        "theta_0, theta_1,feature,prediction",
        [
            [0, 2, 2, 4],
            [2, 2, 2, 6],
            [0, -5, 2, -10],
            [7, -5, 2, -3],
            [17, -5, 2, 7],
            [-5, 2, 2, -1],
        ],
    )
    def test_prediction(self, theta_0, theta_1, feature, prediction):
        assert (
            UnivariateLinearRegressionModel(theta_0, theta_1).predict(feature)
            == prediction
        )


class TestMultivariateLinearRegression:
    @pytest.mark.parametrize(
        "theta_0, gradients,features,prediction",
        [
            [0, Series([2, 2]), Series([4, 8]), 24],
            [5, Series([2, 2]), Series([4, 8]), 29],
            [0, Series([2, -5]), Series([4, 8]), -32],
        ],
    )
    def test_predictions(self, theta_0, gradients, features, prediction):
        assert (
            MultivariateLinearRegression(theta_0, gradients).predict(features)
            == prediction
        )
