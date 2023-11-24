import math
import numpy as np
from scipy.optimize import minimize

# Manhattan distance to Grand River hospital for each of the 13 severe patients
d_gr_manhattan = np.array([9, 7, 3, 8, 3, 8, 14, 15, 8, 7, 10, 12, 15])
# Manhattan distance to St. Mary's hospital for each of the 13 severe patients
d_sm_manhattan = np.array([13, 11, 7, 12, 5, 8, 14, 15, 6, 3, 8, 8, 11])
# Ambulance speed in grid lengths per hour
va = 80
# ERV speed in grid lengths per hour
ve = 50


def objective(x):
    t = 0
    for i in range(len(d_sm_manhattan)):
        # Ambulances at Grand River
        a = (1 / x[0]) * (d_gr_manhattan[i] / va)
        # Ambulances at St. Mary's
        b = (1 / x[1]) * (d_gr_manhattan[i] / va)
        # ERVs at Grand River
        c = (1 / x[2]) * (d_gr_manhattan[i] / ve)
        # ERVs at St. Mary's
        d = (1 / x[3]) * (d_gr_manhattan[i] / ve)
        t += 2 * min(a, b, c, d)
    return - 0.35 * math.e ** -(t / 10)


def constraint(x):
    return [x[0] + x[1] - 48,
            x[2] + x[3] - 8]


bounds = [(0, 48), (0, 48), (0, 8), (0, 8)]

initial_guess = [24, 1, 8, 1]

# Define integer constraints
constraints = ({'type': 'eq', 'fun': constraint})

result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')

# Display the results
print("Optimal values:", result.x)
print("Optimal objective value:", result.fun)
