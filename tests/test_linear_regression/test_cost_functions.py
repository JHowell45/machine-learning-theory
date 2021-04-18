"""

"""
import pytest
from pandas import Series

from src.linear_regression.cost_functions import mean_squared_error


class TestMeanSquaredError:
    @pytest.mark.parametrize(
        "actual,predictions,result",
        [
            [Series([1, 2, 3, 4]), Series([1, 2, 3, 4]), 0],
            [Series([1, 2, 3, 4]), Series([2, 4, 6, 8]), 3.75],
            [Series([1, 2, 3, 4]), Series([5, 5, 5, 5]), 3.75],
            [Series([1, 2, 3, 4]), Series([10, 5, 3, 2]), 11.75],
            [Series([1, 2, 3, 4]), Series([1, 1, 1, 1]), 1.75],
            [Series([1, 2, 3, 4]), Series([324, 4321, 23, 12345]), 21382346.375],
        ],
    )
    def test_mean_squared_error(
        self, actual: Series, predictions: Series, result: float
    ):
        assert mean_squared_error(actual, predictions) == result
