"""

"""
from pandas import Series


def mean_squared_error(actual_labels: Series, predicted_labels: Series) -> float:
    actual_labels_length = len(actual_labels)
    predicted_labels_length = len(predicted_labels)
    if actual_labels_length != predicted_labels_length:
        raise ValueError(
            f"Different lengths for actual labels ({actual_labels_length}) and "
            f"predicted labels ({predicted_labels_length})"
        )
    distance_sum = sum((predicted_labels - actual_labels) ** 2)
    return (1 / (2 * actual_labels_length)) * distance_sum
