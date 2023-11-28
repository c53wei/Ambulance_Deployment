import pytest

import numpy as np

from objective_func import optimize_all


def test_basic(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
               bounds, initial_guess, benchmark):
    benchmark(optimize_all, d_gr_severe=d_gr_severe, d_gr_moderate=d_gr_moderate, d_gr_mild=d_gr_mild,
              d_sm_severe=d_sm_severe, d_sm_moderate=d_sm_moderate, d_sm_mild=d_sm_mild,
              bounds=bounds, initial_guess=initial_guess)


def test_input_10(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                  bounds, initial_guess, benchmark):
    d_gr_severe = np.array(d_gr_severe.tolist() * 10)
    d_gr_moderate = np.array(d_gr_moderate.tolist() * 10)
    d_gr_mild = np.array(d_gr_mild.tolist() * 10)

    d_sm_severe = np.array(d_sm_severe.tolist() * 10)
    d_sm_moderate = np.array(d_sm_moderate.tolist() * 10)
    d_sm_mild = np.array(d_sm_mild.tolist() * 10)

    benchmark(optimize_all, d_gr_severe=d_gr_severe, d_gr_moderate=d_gr_moderate, d_gr_mild=d_gr_mild,
              d_sm_severe=d_sm_severe, d_sm_moderate=d_sm_moderate, d_sm_mild=d_sm_mild,
              bounds=bounds, initial_guess=initial_guess)


def test_input_10(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                  bounds, initial_guess, benchmark):
    multiplier = 10

    d_gr_severe = np.array(d_gr_severe.tolist() * multiplier)
    d_gr_moderate = np.array(d_gr_moderate.tolist() * multiplier)
    d_gr_mild = np.array(d_gr_mild.tolist() * multiplier)

    d_sm_severe = np.array(d_sm_severe.tolist() * multiplier)
    d_sm_moderate = np.array(d_sm_moderate.tolist() * multiplier)
    d_sm_mild = np.array(d_sm_mild.tolist() * multiplier)

    benchmark(optimize_all, d_gr_severe=d_gr_severe, d_gr_moderate=d_gr_moderate, d_gr_mild=d_gr_mild,
              d_sm_severe=d_sm_severe, d_sm_moderate=d_sm_moderate, d_sm_mild=d_sm_mild,
              bounds=bounds, initial_guess=initial_guess)


def test_input_100(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                   bounds, initial_guess, benchmark):
    multiplier = 100

    d_gr_severe = np.array(d_gr_severe.tolist() * multiplier)
    d_gr_moderate = np.array(d_gr_moderate.tolist() * multiplier)
    d_gr_mild = np.array(d_gr_mild.tolist() * multiplier)

    d_sm_severe = np.array(d_sm_severe.tolist() * multiplier)
    d_sm_moderate = np.array(d_sm_moderate.tolist() * multiplier)
    d_sm_mild = np.array(d_sm_mild.tolist() * multiplier)

    benchmark(optimize_all, d_gr_severe=d_gr_severe, d_gr_moderate=d_gr_moderate, d_gr_mild=d_gr_mild,
              d_sm_severe=d_sm_severe, d_sm_moderate=d_sm_moderate, d_sm_mild=d_sm_mild,
              bounds=bounds, initial_guess=initial_guess)


def test_input_500(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                   bounds, initial_guess, benchmark):
    multiplier = 500

    d_gr_severe = np.array(d_gr_severe.tolist() * multiplier)
    d_gr_moderate = np.array(d_gr_moderate.tolist() * multiplier)
    d_gr_mild = np.array(d_gr_mild.tolist() * multiplier)

    d_sm_severe = np.array(d_sm_severe.tolist() * multiplier)
    d_sm_moderate = np.array(d_sm_moderate.tolist() * multiplier)
    d_sm_mild = np.array(d_sm_mild.tolist() * multiplier)

    benchmark(optimize_all, d_gr_severe=d_gr_severe, d_gr_moderate=d_gr_moderate, d_gr_mild=d_gr_mild,
              d_sm_severe=d_sm_severe, d_sm_moderate=d_sm_moderate, d_sm_mild=d_sm_mild,
              bounds=bounds, initial_guess=initial_guess)


def test_input_1000(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                    bounds, initial_guess, benchmark):
    multiplier = 1000

    d_gr_severe = np.array(d_gr_severe.tolist() * multiplier)
    d_gr_moderate = np.array(d_gr_moderate.tolist() * multiplier)
    d_gr_mild = np.array(d_gr_mild.tolist() * multiplier)

    d_sm_severe = np.array(d_sm_severe.tolist() * multiplier)
    d_sm_moderate = np.array(d_sm_moderate.tolist() * multiplier)
    d_sm_mild = np.array(d_sm_mild.tolist() * multiplier)

    benchmark(optimize_all, d_gr_severe=d_gr_severe, d_gr_moderate=d_gr_moderate, d_gr_mild=d_gr_mild,
              d_sm_severe=d_sm_severe, d_sm_moderate=d_sm_moderate, d_sm_mild=d_sm_mild,
              bounds=bounds, initial_guess=initial_guess)


def test_input_5000(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                    bounds, initial_guess, benchmark):
    multiplier = 5000

    d_gr_severe = np.array(d_gr_severe.tolist() * multiplier)
    d_gr_moderate = np.array(d_gr_moderate.tolist() * multiplier)
    d_gr_mild = np.array(d_gr_mild.tolist() * multiplier)

    d_sm_severe = np.array(d_sm_severe.tolist() * multiplier)
    d_sm_moderate = np.array(d_sm_moderate.tolist() * multiplier)
    d_sm_mild = np.array(d_sm_mild.tolist() * multiplier)

    benchmark(optimize_all, d_gr_severe=d_gr_severe, d_gr_moderate=d_gr_moderate, d_gr_mild=d_gr_mild,
              d_sm_severe=d_sm_severe, d_sm_moderate=d_sm_moderate, d_sm_mild=d_sm_mild,
              bounds=bounds, initial_guess=initial_guess)


def test_input_10000(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                     bounds, initial_guess, benchmark):
    multiplier = 10000

    d_gr_severe = np.array(d_gr_severe.tolist() * multiplier)
    d_gr_moderate = np.array(d_gr_moderate.tolist() * multiplier)
    d_gr_mild = np.array(d_gr_mild.tolist() * multiplier)

    d_sm_severe = np.array(d_sm_severe.tolist() * multiplier)
    d_sm_moderate = np.array(d_sm_moderate.tolist() * multiplier)
    d_sm_mild = np.array(d_sm_mild.tolist() * multiplier)

    benchmark(optimize_all, d_gr_severe=d_gr_severe, d_gr_moderate=d_gr_moderate, d_gr_mild=d_gr_mild,
              d_sm_severe=d_sm_severe, d_sm_moderate=d_sm_moderate, d_sm_mild=d_sm_mild,
              bounds=bounds, initial_guess=initial_guess)
