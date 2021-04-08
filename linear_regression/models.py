from dataclasses import dataclass, field
from pandas import Series, DataFrame
from typing import List, Union


@dataclass
class UnivariateLinearRegressionModel:
    gradient: float
    y_axis_shift: float = 0

    def predict(self, feature: float) -> float:
        return (self.gradient * feature) + self.y_axis_shift

    def multiple_predictions(self, features: Union[Series, List[float]]) -> Series:
        """Use this function to quickly predict the values for multiple features.

        This function is used for running the linear regression to get the
        predictions for several features at once.

        :param features: the vector of feature values.
        :return: the vector of feature predictions.
        """
        if isinstance(features, list):
            features = Series(features)
        features_dataframe = DataFrame(
            [Series(1 for _ in range(len(features))), features]
        ).transpose()
        params = Series([self.y_axis_shift, self.gradient])
        return features_dataframe.dot(params)


@dataclass
class MultivariateLinearRegression:
    gradients: Series = field(default_factory=Series)
    y_axis_shift: float = 0

    def predict(self, features: Series) -> float:
        return self.gradients.dot(features) + self.y_axis_shift
