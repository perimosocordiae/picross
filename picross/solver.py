from __future__ import absolute_import
import numpy as np
import warnings

from ._util import valid_gaps

__all__ = ['solve', 'FULL', 'EMPTY']

FULL = 2
EMPTY = 1


def solve(rows, cols):
  nr = len(rows)
  nc = len(cols)
  puzzle = np.zeros((nr, nc), dtype=int)
  missing_rows = np.arange(nr)
  missing_cols = np.arange(nc)
  old_puzzle = puzzle.copy()
  while len(missing_rows) + len(missing_cols) > 0:
    for i in missing_rows:
      puzzle[i] = _solve_1d(puzzle[i], rows[i])
    for i in missing_cols:
      puzzle[:,i] = _solve_1d(puzzle[:,i], cols[i])
    if np.array_equal(puzzle, old_puzzle):
      warnings.warn("Can't solve puzzle: no progress made")
      return puzzle
    old_puzzle[:] = puzzle
    mask = puzzle == 0
    missing_rows, = np.where(mask.any(axis=1))
    missing_cols, = np.where(mask.any(axis=0))
  return puzzle


def _solve_1d(known, constraints):
  nk = len(known)
  nc = len(constraints)
  num_full = sum(constraints)
  num_empty = nk - num_full

  new_known = None
  for gaps in valid_gaps(nc, num_empty):
    # what would the array look like given these gaps?
    arr = _candidate_array(gaps, constraints, known)
    if arr is None:
      # the array would have violated our known values
      continue
    # accumulate common values across all candidates
    if new_known is None:
      new_known = arr
    else:
      new_known = np.bitwise_and(new_known, arr)
  if new_known is None:
    raise Exception('Unsolvable constraint: %s' % constraints)
  return new_known


def _candidate_array(gaps, constraints, known):
  arr = np.zeros_like(known)
  i = 0
  for g, c in zip(gaps, constraints):
    if (known[i:i+g] == FULL).any():
      return None
    arr[i:i+g] = EMPTY
    i += g
    if (known[i:i+c] == EMPTY).any():
      return None
    arr[i:i+c] = FULL
    i += c
  if (known[i:] == FULL).any():
    return None
  arr[i:] = EMPTY
  return arr
