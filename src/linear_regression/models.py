from dataclasses import dataclass, field
from math import inf

from pandas import DataFrame, Series
from tqdm import tqdm

from .parameter_learning.univariate import batch_gradient_descent


@dataclass
class UnivariateLinearRegressionModel:
    theta_0: float
    theta_1: float
    cost_function_score: float = None

    def __post_init__(self):
        self.params = Series([self.theta_0, self.theta_1])

    def predict(self, feature: float) -> float:
        return (self.theta_1 * feature) + self.theta_0

    def multiple_predictions(self, features: Series) -> Series:
        """Use this function to quickly predict the values for multiple features.

        This function is used for running the linear regression to get the predictions for several features at once.

        :param features: the vector of feature values.
        :return: the vector of feature predictions.
        """
        features_dataframe = DataFrame(
            [Series(1 for _ in range(len(features))), features]
        ).transpose()
        return features_dataframe.dot(self.params)

    @classmethod
    def from_gradient_descent(
        cls,
        features: Series,
        labels: Series,
        current_theta_0: int = 0,
        current_theta_1: int = 0,
        learning_rate: float = 0.0001,
        epochs: int = None,
    ) -> "UnivariateLinearRegressionModel":
        """Use this function to generate a model using gradient descent to calculate the params.

        This function is used for generating a univariate linear regression model and calculating the best parameters
        for the model based off of the results of running gradient descent.

        :param features: the features used for training the model on using gradient descent.
        :param labels: the labels used for training the model on using gradient descent.
        :param current_theta_0: the starting value of theta 0 for the gradient descent.
        :param current_theta_1: the starting value of theta 1 for the gradient descent.
        :param learning_rate: the rate at which gradient descent is done, smaller is more accurate, but slower.
        :param epochs: the number of times to run gradient descent, good if it takes a long time to run.
        :return: the generated model with calculated parameters.
        """
        params = batch_gradient_descent(
            univariate_linear_regression_model=cls,
            features=features,
            labels=labels,
            current_theta_0=current_theta_0,
            current_theta_1=current_theta_1,
            learning_rate=learning_rate,
            epochs=epochs,
        )
        return cls(
            theta_0=params["current_theta_0"],
            theta_1=params["current_theta_1"],
            cost_function_score=params["current_mse_score"],
        )


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
