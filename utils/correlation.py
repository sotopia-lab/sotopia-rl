import numpy as np


def fleiss_kappa(ratings: np.ndarray) -> float:
    n, k = ratings.shape  # n is number of items, k is number of raters
    N = np.sum(
        ratings[0]
    )  # Total number of ratings per item (assumed to be the same for all items)

    # Calculating p_i
    p_i = np.sum(ratings * (ratings - 1), axis=1) / (N * (N - 1))
    P_o = np.mean(p_i)  # Mean of p_i

    # Calculating p_e
    p = np.sum(ratings, axis=0) / (
        n * N
    )  # Sum over items, divide by total number of ratings
    P_e = np.sum(p**2)

    # Calculating kappa
    kappa = (P_o - P_e) / (1 - P_e)
    return kappa
