"""

"""
from numpy.linalg import pinv
from pandas import DataFrame, Series


def normal_equation(features: DataFrame, labels: Series) -> Series:
    return (
        pinv(features.transpose().mul(features)).mul(features.transpose()).mul(labels)
    )
