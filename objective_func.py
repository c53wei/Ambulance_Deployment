import math
import numpy as np

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

# 1 grid length = .5 km
# 30 L / 100 km => .15 L / grid length => .15 L/grid length *  .0150 $ / L => 0.00225 $/grid length
c_a = .00225
# Dollars to drive per grid distance for the ERV let's say 2016 ford explorer
# 14.9 L / 100 km ==> .0745 L / grid length => .0011175 $ / grid length
c_e = 0.0011175


def survival_exp(x, d_gr, d_sm, coefficient, denominator):
    # Ambulance speed in grid lengths per hour
    v_a = 160
    # ERV speed in grid lengths per hour
    v_e = 100

    t = 0
    for i in range(len(d_gr)):
        # Ambulances at Grand River
        a = (1 / x[0]) * (d_gr[i] / v_a)
        # Ambulances at St. Mary's
        b = (1 / x[1]) * (d_sm[i] / v_a)
        # ERVs at Grand River
        c = (1 / x[2]) * (d_gr[i] / v_e)
        # ERVs at St. Mary's
        d = (1 / x[3]) * (d_sm[i] / v_e)
        t += 2 * (a + b + c + d)
    # return t
    return - coefficient * math.e ** -(t / denominator)


def survival_step(x, d_gr, d_sm):
    # Ambulance speed in grid lengths per hour
    v_a = 160
    # ERV speed in grid lengths per hour
    v_e = 100

    t = 0
    for i in range(len(d_gr)):
        # Ambulances at Grand River
        a = (1 / x[0]) * (d_gr[i] / v_a)
        # Ambulances at St. Mary's
        b = (1 / x[1]) * (d_sm[i] / v_a)
        # ERVs at Grand River
        c = (1 / x[2]) * (d_gr[i] / v_e)
        # ERVs at St. Mary's
        d = (1 / x[3]) * (d_sm[i] / v_e)
        t += 2 * (a + b + c + d)
    return -1 if t < 8 else 0


def constraint(x):
    return [-x[0] - x[1] + 48,
            -x[2] - x[3] + 8]



def maximize_survival_exp(d_gr, d_sm, x0, coefficient, denominator, bounds):
    """Handles nonlinear equality and inequality constraints.
    Suitable for constrained optimization problems."""
    constraints = ({'type': 'ineq', 'fun': constraint})
    return minimize(survival_exp, x0, args=(d_gr, d_sm, coefficient, denominator),
                    bounds=bounds, constraints=constraints, method='trust-constr')


def maximize_survival_step(d_gr, d_sm, x0, bounds):
    """Suitable for constrained optimization problems.
       Efficient for problems with smooth and differentiable objective functions."""
    constraints = ({'type': 'eq', 'fun': constraint})
    return minimize(survival_step, x0, args=(d_gr, d_sm),
                    bounds=bounds, constraints=constraints, method='SLSQP')


def minimize_fuel_cost(d_gr, d_sm):
    d_gr_total_sum = d_gr.sum()
    d_sm_total_sum = d_sm.sum()
    w1 = 2 * c_a * d_gr_total_sum
    w2 = 2 * c_a * d_sm_total_sum
    w3 = 2 * c_e * d_gr_total_sum
    w4 = 2 * c_e * d_sm_total_sum
    c = np.array([w1, w2, w3, w4])
    A = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 1]])
    b_u = np.array([48, 8])
    b_l = np.array([48, 8])

    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones_like(c)
    return milp(c=c, constraints=constraints, integrality=integrality)


def optimize_all(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                 bounds, initial_guess):
    """
    Wrap all objective optimizations for benchmarking
    :param d_gr_severe:
    :param d_gr_moderate:
    :param d_gr_mild:
    :param d_sm_severe:
    :param d_sm_moderate:
    :param d_sm_mild:
    :param bounds:
    :param initial_guess:
    :return:
    """
    maximize_survival_exp(d_gr=d_gr_severe, d_sm=d_sm_severe, x0=initial_guess,
                          coefficient=0.35, denominator=10, bounds=bounds)

    maximize_survival_exp(d_gr=d_gr_moderate, d_sm=d_sm_moderate,
                          x0=initial_guess, coefficient=0.5, denominator=4, bounds=bounds)

    maximize_survival_step(d_gr=d_gr_mild, d_sm=d_sm_mild, x0=initial_guess,
                           bounds=bounds)
    d_gr_total = np.concatenate([d_gr_severe, d_gr_moderate, d_gr_mild])
    # Manhattan distance to St. Mary's hospital for all patients
    d_sm_total = np.concatenate([d_sm_severe, d_sm_moderate, d_sm_mild])
    minimize_fuel_cost(d_gr=d_gr_total, d_sm=d_sm_total)


if __name__ == '__main__':
    bounds = [(0, 48), (0, 48), (0, 8), (0, 8)]
    initial_guess = np.array([1, 1, 1, 1])

    # Manhattan distance to Grand River hospital for each of the 13 severe patients
    d_gr_severe_manhattan = np.array([9, 7, 3, 8, 3, 8, 14, 15, 8, 7, 10, 12, 15])
    # Manhattan distance to St. Mary's hospital for each of the 13 severe patients
    d_sm_severe_manhattan = np.array([13, 11, 7, 12, 5, 8, 14, 15, 6, 3, 8, 8, 11])
    severe_result = maximize_survival_exp(d_gr=d_gr_severe_manhattan, d_sm=d_sm_severe_manhattan, x0=initial_guess,
                                          coefficient=0.35, denominator=10, bounds=bounds)
    # Display the results
    print(f"Optimal values: {severe_result.x[0]:.0f}, {severe_result.x[1]:.0f}, "
          f"{severe_result.x[2]:.0f}, {severe_result.x[3]:.0f}")
    print(f"Optimal objective value: {severe_result.fun:.5f}")

    # Manhattan distance to Grand River hospital for each of the 14 moderate patients
    d_gr_moderate_manhattan = np.array([8, 7, 6, 4, 3, 7, 13, 14, 2, 9, 7, 6, 9, 13])
    # Manhattan distance to St. Mary's hospital for each of the 14 moderate patients
    d_sm_moderate_manhattan = np.array([12, 11, 10, 6, 7, 11, 15, 16, 4, 9, 5, 2, 7, 9])
    moderate_result = maximize_survival_exp(d_gr=d_gr_moderate_manhattan, d_sm=d_sm_moderate_manhattan,
                                            x0=initial_guess, coefficient=0.5, denominator=4, bounds=bounds)
    # Display the results
    print(f"Optimal values: {moderate_result.x[0]:.0f}, {moderate_result.x[1]:.0f}, "
          f"{moderate_result.x[2]:.0f}, {moderate_result.x[3]:.0f}")
    print(f"Optimal objective value: {moderate_result.fun:.5f}")

    # Manhattan distance to Grand River hospital for each of the 8 mild patients
    d_gr_mild_manhattan = np.array([10, 6, 4, 9, 7, 9, 8, 11])
    # Manhattan distance to St. Mary's hospital for each of the 8 mild patients
    d_sm_mild_manhattan = np.array([14, 10, 8, 13, 7, 7, 6, 7])
    mild_result = maximize_survival_step(d_gr=d_gr_mild_manhattan, d_sm=d_sm_mild_manhattan, x0=initial_guess,
                                         bounds=bounds)
    # Display the results
    print(f"Optimal values: {mild_result.x[0]:.0f}, {mild_result.x[1]:.0f}, "
          f"{mild_result.x[2]:.0f}, {mild_result.x[3]:.0f}")
    print(f"Optimal objective value: {mild_result.fun:.5f}")

    # Manhattan distance to Grand River hospital for all patients
    d_gr_total_manhattan = np.concatenate([d_gr_severe_manhattan, d_gr_moderate_manhattan, d_gr_mild_manhattan])
    # Manhattan distance to St. Mary's hospital for all patients
    d_sm_total_manhattan = np.concatenate([d_sm_severe_manhattan, d_sm_moderate_manhattan, d_sm_mild_manhattan])

    fuel_cost_result = minimize_fuel_cost(d_gr=d_gr_total_manhattan, d_sm=d_sm_total_manhattan)

    # Display the results
    print(f"Optimal values: {fuel_cost_result.x[0]:.0f}, {fuel_cost_result.x[1]:.0f}, "
          f"{fuel_cost_result.x[2]:.0f}, {fuel_cost_result.x[3]:.0f}")
    print(f"Optimal objective value: {fuel_cost_result.fun:.5f}")

    optimize_all(d_gr_severe=d_gr_severe_manhattan, d_gr_moderate=d_gr_moderate_manhattan,
                 d_gr_mild=d_gr_mild_manhattan, d_sm_severe=d_sm_severe_manhattan,
                 d_sm_moderate=d_sm_moderate_manhattan, d_sm_mild=d_sm_mild_manhattan,
                 bounds=bounds, initial_guess=initial_guess)
