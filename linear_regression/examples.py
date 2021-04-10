"""

"""
from sklearn.datasets import load_boston
from linear_regression.models import UnivariateLinearRegressionModel
from linear_regression.parameter_learning import ulr_batch_gradient_descent
from math import inf
from matplotlib import pyplot as plt
from time import time
from random import randrange
from pandas import Series


def univariate_linear_regression_example(
    feature_size: int = 100, learning_rate: int = 0.001
):
    print("\nRunning Univariate Linear Regression Example:\n\n")
    features = Series(feature for feature in range(feature_size))
    labels = Series(feature * 2 for feature in features)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    plt.clf()
    plt.plot(features, labels, "bo")
    plt.xlabel("features")
    plt.ylabel("labels")
    plt.savefig(f"./graphs/linear_regression/univariate_example/artificial_data.png")

    s = time()
    best_parameters = ulr_batch_gradient_descent(
        features=features, labels=labels, learning_rate=learning_rate
    )
    print(f"Gradient Descent Runtime: {round(time() - s, 2)}s")
    print(f"Best Parameters: {best_parameters}")
    print(f"Best MSE Score: {best_parameters['current_mse_score']}")
    best_model = UnivariateLinearRegressionModel(
        theta_0=best_parameters["current_theta_0"],
        theta_1=best_parameters["current_theta_1"],
    )
    print()
    return best_model


def univariate_linear_regression_example_2(feature_size: int = 100):
    print("\nRunning Univariate Linear Regression Example:\n\n")
    features = Series(randrange(0, feature_size * 2) for _ in range(feature_size))
    labels = Series(4 + 3 * feature + randrange(0, 10) for feature in features)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    plt.clf()
    plt.plot(features, labels, "bo")
    plt.xlabel("features")
    plt.ylabel("labels")
    plt.savefig(f"./graphs/linear_regression/univariate_example/artificial_data_2.png")

    s = time()
    best_parameters = ulr_batch_gradient_descent(
        features=features, labels=labels, epochs=100000
    )
    print(f"Gradient Descent Runtime: {round(time() - s, 2)}s")
    print(f"Best Parameters: {best_parameters}")
    print(f"Best MSE Score: {best_parameters['current_mse_score']}")
    best_model = UnivariateLinearRegressionModel(
        theta_0=best_parameters["current_theta_0"],
        theta_1=best_parameters["current_theta_1"],
    )
    print()
    return best_model


def univariate_linear_regression_example_3():
    print("\nRunning Univariate Linear Regression Example 2:\n\n")
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

        plt.plot(labels, feature_set, "bo")
        plt.xlabel("Median House Price ($1000s)")
        plt.ylabel(name)
        plt.savefig(f"./graphs/linear_regression/univariate_example/{name}.png")

        best_parameters = ulr_batch_gradient_descent(
            features=feature_set, labels=labels
        )
        print(f"Best Parameters: {best_parameters}")
        if best_parameters["current_mse_score"] < best_mse_score:
            best_mse_score = best_parameters["current_mse_score"]
            best_feature_name = name
            best_model = UnivariateLinearRegressionModel(
                theta_0=best_parameters["current_theta_0"],
                theta_1=best_parameters["current_theta_1"],
            )
        print()
    print(f"Best MSE Score: {best_mse_score}")
    print(f"Best feature for predicting: '{best_feature_name}'")
    return best_model
