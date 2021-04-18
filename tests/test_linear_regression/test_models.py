"""

"""
import pytest

from src.linear_regression.models import UnivariateLinearRegressionModel


class TestUnivariateLinearRegressionModel:
    @pytest.mark.parametrize(
        "theta_0, theta_1,feature,prediction",
        [[0, 2, 2, 4]],
    )
    def test_prediction(self, theta_0, theta_1, feature, prediction):
        assert (
            UnivariateLinearRegressionModel(theta_0, theta_1).predict(feature)
            == prediction
        )
