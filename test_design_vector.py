import pytest

import numpy as np

from n_dimension_design import optimize_all


def test_basic(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
               bounds, initial_guess, benchmark):
    benchmark(optimize_all, d_severe=np.array([d_gr_severe, d_sm_severe]),
              d_moderate=np.array([d_gr_moderate, d_sm_moderate]),
              d_mild=np.array([d_gr_mild, d_sm_mild]),
              bounds=bounds, initial_guess=initial_guess)


def test_design_2(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                  bounds, initial_guess, benchmark):
    multiplier = 2
    d_severe = np.repeat(np.array([d_gr_severe, d_sm_severe]), multiplier, axis=0)
    d_moderate = np.repeat(np.array([d_gr_moderate, d_sm_moderate]), multiplier, axis=0)
    d_mild = np.repeat(np.array([d_gr_mild, d_sm_mild]), multiplier, axis=0)

    bounds = bounds * multiplier
    initial_guess = np.tile(initial_guess, multiplier)

    benchmark(optimize_all, d_severe=d_severe,
              d_moderate=d_moderate,
              d_mild=d_mild,
              bounds=bounds, initial_guess=initial_guess)


def test_design_10(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                   bounds, initial_guess, benchmark):
    multiplier = 10
    d_severe = np.repeat(np.array([d_gr_severe, d_sm_severe]), multiplier, axis=0)
    d_moderate = np.repeat(np.array([d_gr_moderate, d_sm_moderate]), multiplier, axis=0)
    d_mild = np.repeat(np.array([d_gr_mild, d_sm_mild]), multiplier, axis=0)

    bounds = bounds * multiplier
    initial_guess = np.tile(initial_guess, multiplier)

    benchmark(optimize_all, d_severe=d_severe,
              d_moderate=d_moderate,
              d_mild=d_mild,
              bounds=bounds, initial_guess=initial_guess)

def test_design_20(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                    bounds, initial_guess, benchmark):
    multiplier = 20
    d_severe = np.repeat(np.array([d_gr_severe, d_sm_severe]), multiplier, axis=0)
    d_moderate = np.repeat(np.array([d_gr_moderate, d_sm_moderate]), multiplier, axis=0)
    d_mild = np.repeat(np.array([d_gr_mild, d_sm_mild]), multiplier, axis=0)

    bounds = bounds * multiplier
    initial_guess = np.tile(initial_guess, multiplier)

    benchmark(optimize_all, d_severe=d_severe,
              d_moderate=d_moderate,
              d_mild=d_mild,
              bounds=bounds, initial_guess=initial_guess)

def test_design_30(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                    bounds, initial_guess, benchmark):
    multiplier = 30
    d_severe = np.repeat(np.array([d_gr_severe, d_sm_severe]), multiplier, axis=0)
    d_moderate = np.repeat(np.array([d_gr_moderate, d_sm_moderate]), multiplier, axis=0)
    d_mild = np.repeat(np.array([d_gr_mild, d_sm_mild]), multiplier, axis=0)

    bounds = bounds * multiplier
    initial_guess = np.tile(initial_guess, multiplier)

    benchmark(optimize_all, d_severe=d_severe,
              d_moderate=d_moderate,
              d_mild=d_mild,
              bounds=bounds, initial_guess=initial_guess)

def test_design_40(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                    bounds, initial_guess, benchmark):
    multiplier = 40
    d_severe = np.repeat(np.array([d_gr_severe, d_sm_severe]), multiplier, axis=0)
    d_moderate = np.repeat(np.array([d_gr_moderate, d_sm_moderate]), multiplier, axis=0)
    d_mild = np.repeat(np.array([d_gr_mild, d_sm_mild]), multiplier, axis=0)

    bounds = bounds * multiplier
    initial_guess = np.tile(initial_guess, multiplier)

    benchmark(optimize_all, d_severe=d_severe,
              d_moderate=d_moderate,
              d_mild=d_mild,
              bounds=bounds, initial_guess=initial_guess)


def test_design_50(d_gr_severe, d_gr_moderate, d_gr_mild, d_sm_severe, d_sm_moderate, d_sm_mild,
                   bounds, initial_guess, benchmark):
    multiplier = 50
    d_severe = np.repeat(np.array([d_gr_severe, d_sm_severe]), multiplier, axis=0)
    d_moderate = np.repeat(np.array([d_gr_moderate, d_sm_moderate]), multiplier, axis=0)
    d_mild = np.repeat(np.array([d_gr_mild, d_sm_mild]), multiplier, axis=0)

    bounds = bounds * multiplier
    initial_guess = np.tile(initial_guess, multiplier)

    benchmark(optimize_all, d_severe=d_severe,
              d_moderate=d_moderate,
              d_mild=d_mild,
              bounds=bounds, initial_guess=initial_guess)



