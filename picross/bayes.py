from __future__ import division, absolute_import
import numpy as np
import warnings

from ._util import valid_gaps

NEAR_ZERO = 1e-12
NEAR_ONE = 1 - NEAR_ZERO


def iter_solve(rows, cols):
  nr = len(rows)
  nc = len(cols)
  row_counts = np.array(list(map(sum, rows)))
  col_counts = np.array(list(map(sum, cols)))
  num_full = row_counts.sum()
  assert col_counts.sum() == num_full

  # set up prior as a fairly flat normal distribution
  mean = np.full((nr, nc), num_full / (nr * nc))
  var = np.full_like(mean, 1)
  yield mean.copy()

  missing_rows = np.arange(nr)
  missing_cols = np.arange(nc)
  prev_num_unknown = nr * nc
  while True:
    for i in missing_rows:
      _update_row_inplace(mean[i], var[i], rows[i])
      yield mean.copy()

    for i in missing_cols:
      _update_row_inplace(mean[:,i], var[:,i], cols[i])
      yield mean.copy()

    # check convergence
    unknown = var > NEAR_ZERO
    num_unknown = np.count_nonzero(unknown)
    if prev_num_unknown == num_unknown:
      warnings.warn("Can't solve puzzle: no progress made")
      return
    elif num_unknown == 0:
      return

    # setup for next iteration
    prev_num_unknown = num_unknown
    missing_rows, = np.where(unknown.any(axis=1))
    missing_cols, = np.where(unknown.any(axis=0))


def _update_row_inplace(mu, var, constraints):
  obs_mu, obs_var = _observations(mu, constraints)
  # update using bayes rule for normal distributions
  denom = var + obs_var
  mask = denom > NEAR_ZERO
  mu[mask] = (obs_var[mask] * mu[mask] + var[mask] * obs_mu[mask]) / denom[mask]
  var[mask] = var[mask] * obs_var[mask] / denom[mask]


def _observations(prior, constraints):
  num_full = sum(constraints)
  num_empty = len(prior) - num_full

  candidates = []
  for gaps in valid_gaps(len(constraints), num_empty):
    # what would the array look like given these gaps?
    arr = _candidate_array(gaps, constraints, prior)
    if arr is not None:
      candidates.append(arr)
  candidates = np.array(candidates)
  return candidates.mean(axis=0), candidates.var(axis=0)


def _candidate_array(gaps, constraints, known):
  arr = np.zeros_like(known)
  i = 0
  for g, c in zip(gaps, constraints):
    if (known[i:i+g] > NEAR_ONE).any():
      return None
    arr[i:i+g] = 0
    i += g
    if (known[i:i+c] < NEAR_ZERO).any():
      return None
    arr[i:i+c] = 1
    i += c
  if (known[i:] > NEAR_ONE).any():
    return None
  arr[i:] = 0
  return arr
