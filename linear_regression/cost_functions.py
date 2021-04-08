"""

"""
from typing import List, Union


def mean_squared_error(
    actual_labels: List[Union[int, float]], predicted_labels: List[Union[int, float]]
) -> float:
    actual_labels_length = len(actual_labels)
    predicted_labels_length = len(predicted_labels)
    if actual_labels_length != predicted_labels_length:
        raise ValueError(
            f"Different lengths for actual labels ({actual_labels_length}) and "
            f"predicted labels ({predicted_labels_length})"
        )
    distance_sum = 0
    for actual, predicted in zip(actual_labels, predicted_labels):
        distance_sum += (predicted - actual) ** 2
    return (1 / (2 * actual_labels_length)) * distance_sum
