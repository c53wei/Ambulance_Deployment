import pytest

import numpy as np

@pytest.fixture()
def d_gr_severe():
    # Manhattan distance to Grand River hospital for each of the 13 severe patients
    return np.array([9, 7, 3, 8, 3, 8, 14, 15, 8, 7, 10, 12, 15])


@pytest.fixture()
def d_sm_severe():
    # Manhattan distance to St. Mary's hospital for each of the 13 severe patients
    return np.array([13, 11, 7, 12, 5, 8, 14, 15, 6, 3, 8, 8, 11])


@pytest.fixture()
def d_gr_moderate():
    # Manhattan distance to Grand River hospital for each of the 14 moderate patients
    return np.array([8, 7, 6, 4, 3, 7, 13, 14, 2, 9, 7, 6, 9, 13])


@pytest.fixture()
def d_sm_moderate():
    # Manhattan distance to St. Mary's hospital for each of the 14 moderate patients
    return np.array([12, 11, 10, 6, 7, 11, 15, 16, 4, 9, 5, 2, 7, 9])


@pytest.fixture()
def d_gr_mild():
    # Manhattan distance to Grand River hospital for each of the 8 mild patients
    return np.array([10, 6, 4, 9, 7, 9, 8, 11])


@pytest.fixture()
def d_sm_mild():
    # Manhattan distance to St. Mary's hospital for each of the 8 mild patients
    return np.array([14, 10, 8, 13, 7, 7, 6, 7])


@pytest.fixture()
def bounds():
    return [(0, 48), (0, 48), (0, 8), (0, 8)]


@pytest.fixture()
def initial_guess():
    return np.array([1, 1, 1, 1])
