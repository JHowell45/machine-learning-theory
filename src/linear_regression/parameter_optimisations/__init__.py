"""

"""
from numpy.linalg import pinv
from pandas import DataFrame, Series


def normal_equation(
    features: DataFrame, labels: Series, add_theta_zero: bool = False
) -> Series:
    if add_theta_zero:
        features.insert(0, "x0", [1 for _ in range(features.shape[0])])
    return Series(
        pinv(features.transpose().dot(features)).dot(features.transpose()).dot(labels)
    )
