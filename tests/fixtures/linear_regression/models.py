"""

"""
import pytest

from src.linear_regression.models import UnivariateLinearRegressionModel


@pytest.fixture(scope="session")
def univariate_linear_regression_model(theta_0, theta_1):
    return UnivariateLinearRegressionModel(theta_0, theta_1)
