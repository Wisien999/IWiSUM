import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    mean = np.mean(train_data)
    std = np.std(train_data)
    lower_bound = mean - 3.5 * std
    upper_bound = mean + 2.5 * std
    predictions = np.where((test_data > lower_bound) & (test_data < upper_bound), 0, 1)

    return predictions.tolist()
