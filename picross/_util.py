from six.moves import xrange


def valid_gaps(num_constraints, num_empty):
  for gaps in _sum_combinations(num_constraints+1, num_empty):
    # make sure there's at least a gap of 1 between full sections
    if len(gaps) > 2 and not all(gaps[1:-1]):
      continue
    yield gaps


def _sum_combinations(n, s):
  '''Generate all lists of length n that sum to s.'''
  if n == 1:
    yield [s]
  else:
    for i in xrange(s+1):
      for j in _sum_combinations(n-1, s-i):
        yield [i] + j
