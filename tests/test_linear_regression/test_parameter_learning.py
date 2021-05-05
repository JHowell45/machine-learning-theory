"""

"""
from pandas import DataFrame, Series

from src.linear_regression.parameter_optimisations import normal_equation

label_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

features = DataFrame(
    {
        "theta_zero": [1 for _ in label_values],
        "feature_1": [x / 10 for x in label_values],
    }
)

labels = Series(label_values)


class TestNormalEquation:

    results = normal_equation(features, labels)

    def test_type(self):
        assert isinstance(self.results, Series)

    def test_length(self):
        assert len(self.results) == features.shape[1]

    def test_value(self):
        results = [round(x, 2) for x in self.results]
        assert results == [0, 10]
