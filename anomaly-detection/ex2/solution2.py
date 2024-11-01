import numpy as np
from sklearn.covariance import MinCovDet


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    covs = MinCovDet().fit(train_data)
    mahalanobis = covs.mahalanobis(train_data)
    maximum = np.max(mahalanobis)

    result = np.where(covs.mahalanobis(test_data) <= maximum, 0, 1)

    return result.tolist()
