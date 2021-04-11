from dataclasses import dataclass, field
from pandas import Series, DataFrame
from typing import List, Union


@dataclass
class UnivariateLinearRegressionModel:
    theta_0: float
    theta_1: float

    def __post_init__(self):
        self.params = Series([self.theta_0, self.theta_1])

    def predict(self, feature: float) -> float:
        return (self.theta_1 * feature) + self.theta_0

    def multiple_predictions(self, features: Series) -> Series:
        """Use this function to quickly predict the values for multiple features.

        This function is used for running the linear regression to get the
        predictions for several features at once.

        :param features: the vector of feature values.
        :return: the vector of feature predictions.
        """
        features_dataframe = DataFrame(
            [Series(1 for _ in range(len(features))), features]
        ).transpose()
        return features_dataframe.dot(self.params)


@dataclass
class MultivariateLinearRegression:
    theta_0: float
    gradients: Series = field(default_factory=Series)

    @property
    def params(self):
        return Series([self.theta_0]).append(self.gradients, ignore_index=True)

    def predict(self, features: Series) -> float:
        if len(self.gradients) != len(features):
            raise ValueError(
                f"Features not the same length as gradients! Features: "
                f"{len(features)}, Gradients: {len(self.gradients)}"
            )
        return self.gradients.multiply(features).sum() + self.theta_0

    def multiple_predictions(self, features: DataFrame) -> Series:
        rows, columns = features.shape
        if len(self.gradients) != columns:
            raise ValueError(
                f"Features not the same length as gradients! Features: {columns}, Gradients: {len(self.gradients)}"
            )
        return features.mul(self.gradients).sum(1).add(self.theta_0)
