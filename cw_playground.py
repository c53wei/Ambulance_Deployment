import math
import numpy as np
from scipy.optimize import minimize


def survival_exp(x, d_gr, d_sm, coefficient):
    # Ambulance speed in grid lengths per hour
    va = 80
    # ERV speed in grid lengths per hour
    ve = 50

    t = 0
    for i in range(len(d_gr)):
        # Ambulances at Grand River
        a = (1 / x[0]) * (d_gr[i] / va)
        # Ambulances at St. Mary's
        b = (1 / x[1]) * (d_sm[i] / va)
        # ERVs at Grand River
        c = (1 / x[2]) * (d_gr[i] / ve)
        # ERVs at St. Mary's
        d = (1 / x[3]) * (d_sm[i] / ve)
        t += 2 * (a + b + c + d)
    return - coefficient * math.e ** -(t / 10)


def survival_step(x, d_gr, d_sm):
    # Ambulance speed in grid lengths per hour
    va = 80
    # ERV speed in grid lengths per hour
    ve = 50

    t = 0
    for i in range(len(d_gr)):
        # Ambulances at Grand River
        a = (1 / x[0]) * (d_gr[i] / va)
        # Ambulances at St. Mary's
        b = (1 / x[1]) * (d_sm[i] / va)
        # ERVs at Grand River
        c = (1 / x[2]) * (d_gr[i] / ve)
        # ERVs at St. Mary's
        d = (1 / x[3]) * (d_sm[i] / ve)
        t += 2 * (a + b + c + d)
    return -1 if t < 8 else 0


def constraint(x):
    return [-x[0] - x[1] + 48,
            -x[2] - x[3] + 8]


bounds = [(0, 48), (0, 48), (0, 8), (0, 8)]
initial_guess = [32, 1, 3, 1]
# Define integer constraints
constraints = ({'type': 'ineq', 'fun': constraint})

# Manhattan distance to Grand River hospital for each of the 13 severe patients
d_gr_manhattan = np.array([9, 7, 3, 8, 3, 8, 14, 15, 8, 7, 10, 12, 15])
# Manhattan distance to St. Mary's hospital for each of the 13 severe patients
d_sm_manhattan = np.array([13, 11, 7, 12, 5, 8, 14, 15, 6, 3, 8, 8, 11])
severe_result = minimize(survival_exp, initial_guess, args=(d_gr_manhattan, d_sm_manhattan, 0.35),
                         bounds=bounds, constraints=constraints, method='trust-constr')
# Display the results
print(f"Optimal values: {severe_result.x[0]:.0f}, {severe_result.x[1]:.0f}, "
      f"{severe_result.x[2]:.0f}, {severe_result.x[3]:.0f}")
print(f"Optimal objective value: {severe_result.fun:.5f}")
# Manhattan distance to Grand River hospital for each of the 14 moderate patients
d_gr_manhattan = np.array([8, 7, 6, 4, 3, 7, 13, 14, 2, 9, 7, 6, 9, 13])
# Manhattan distance to St. Mary's hospital for each of the 13 severe patients
d_sm_manhattan = np.array([12, 11, 10, 6, 7, 11, 15, 16, 4, 9, 5, 2, 7, 9])
moderate_result = minimize(survival_exp, initial_guess, args=(d_gr_manhattan, d_sm_manhattan, 0.5),
                           bounds=bounds, constraints=constraints, method='trust-constr')
# Display the results
print(f"Optimal values: {moderate_result.x[0]:.0f}, {moderate_result.x[1]:.0f}, "
      f"{moderate_result.x[2]:.0f}, {moderate_result.x[3]:.0f}")
print(f"Optimal objective value: {moderate_result.fun:.5f}")

