"""

"""
from sklearn.datasets import load_boston
from linear_regression.models import UnivariateLinearRegressionModel
from linear_regression.parameter_learning import ulr_batch_gradient_descent
from math import inf
from matplotlib import pyplot as plt


def univariate_linear_regression_example():
    print("\nRunning Univariate Linear Regression Example:\n\n")
    features, labels = load_boston(return_X_y=True)
    print("Loading example Boston data from SKLearn:\n")
    print(f"Features Shape: {features.shape}")
    print(f"Labels Shape: {labels.shape}")
    print()
    feature_names = [
        "crime_per_capita",
        "zn_proportion_of_residential_land",
        "indus_proportion_of_non_retail",
        "chas_charles_river_dummy_var",
        "nox_concentration",
        "average_no_of_rooms",
        "proportion_of_occupied_homes_built_before_1940",
        "weighted_distance_between_5_boston_employment_centres",
        "index_of_accessibility_to_highways",
        "property_tax_per_10K",
        "pupil_teacher_ration_by_town",
        "racial_diversity",
        "lstat",
    ]
    best_mse_score = inf
    best_feature_name = ""
    best_model = None
    for name, feature_set in zip(feature_names, features.transpose()):
        print(f"'{name}':")

        plt.plot(feature_set, labels, "bo")
        plt.xlabel(name)
        plt.ylabel("Median House Price ($1000s)")
        plt.savefig(f"./graphs/linear_regression/univariate_example/{name}.png")

        best_parameters = ulr_batch_gradient_descent(
            features=feature_set, labels=labels
        )
        print(f"Best Parameters: {best_parameters}")
        if best_parameters["current_score"] < best_mse_score:
            best_mse_score = best_parameters["current_score"]
            best_feature_name = name
            best_model = UnivariateLinearRegressionModel(
                gradient=best_parameters["current_gradient"],
                y_axis_shift=best_parameters["current_y_shift"],
            )
        print()
    print(f"Best MSE Score: {best_mse_score}")
    print(f"Best feature for predicting: '{best_feature_name}'")
    return best_model
