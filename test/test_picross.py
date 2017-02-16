import unittest
import numpy as np
from numpy.testing import assert_array_equal

from picross import solve, FULL as X, EMPTY as O


class PicrossTest(unittest.TestCase):
  def test_solve_1(self):
    rows = [[2,1,1,1,1,1,1],[5,1,1,4],[4,4],[1,1],[3],
            [7],[6],[3],[1,3],[2,4],
            [3,8],[13],[12],[12],[15],
            [6,2,1],[6,1,2,1],[2,1,1,2,3,1,1],[2,2,1,4,1,2,1],[5,1,2,1,2,1,2]]
    cols = [[2,4],[4,6],[7,1],[1,9],[2,6,2],
            [1,6],[2,6,3],[1,5],[3,5,2],[1,1,10],
            [1,2,7,2],[15,2],[11],[15,3],[1,2,1,2,1],
            [1,2,4],[3,2],[1,3],[2,1],[1,3]]
    expected = np.array([
        [O, O, O, X, X, O, X, O, X, O, O, X, O, X, O, O, X, O, X, O],
        [O, O, O, O, X, X, X, X, X, O, O, X, O, X, O, O, X, X, X, X],
        [O, O, O, O, O, O, O, O, X, X, X, X, O, X, X, X, X, O, O, O],
        [O, O, O, O, O, O, O, O, O, O, O, X, O, X, O, O, O, O, O, O],
        [O, O, O, O, O, O, O, O, O, O, O, X, X, X, O, O, O, O, O, O],
        [O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, O, O, O, O],
        [O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, O, O, O, O],
        [O, O, O, O, O, O, O, O, O, O, O, X, X, X, O, O, O, O, O, O],
        [O, X, O, O, O, O, O, O, O, O, O, X, X, X, O, O, O, O, O, O],
        [X, X, O, O, O, O, O, O, O, O, X, X, X, X, O, O, O, O, O, O],
        [X, X, X, O, O, O, X, X, X, X, X, X, X, X, O, O, O, O, O, O],
        [O, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O],
        [O, O, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O],
        [O, O, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O],
        [O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O],
        [O, X, X, X, X, X, X, O, O, X, X, O, O, O, O, X, O, O, O, O],
        [X, X, X, X, X, X, O, O, O, X, O, O, O, O, X, X, O, X, O, O],
        [X, X, O, X, O, O, X, O, O, X, X, O, O, X, X, X, O, X, O, X],
        [X, X, O, X, X, O, X, O, X, X, X, X, O, X, O, O, X, X, O, X],
        [X, X, X, X, X, O, X, O, X, X, O, X, O, X, X, O, X, O, X, X],
    ], dtype=int)
    soln = solve(rows, cols)
    assert_array_equal(soln, expected)

  def test_solve_2(self):
    rows = [[1],[4,3],[3,5],[2,2,5],[3,3,7],
            [8,7],[6,6],[6,6],[5,5],[11],
            [7],[9],[10],[10],[10],
            [7],[14],[14],[14],[12]]
    cols = [[3],[3],[2],[3,3],[5,4],
            [1,6,3,4],[10,9],[3,15],[2,14],[11],
            [11],[3,11],[16],[11,4],[6,4,4],
            [8,4],[8,3],[5],[4],[2]]
    expected = np.array([
        [O, O, O, O, O, O, X, O, O, O, O, O, O, O, O, O, O, O, O, O],
        [O, O, O, O, O, X, X, X, X, O, O, O, O, O, O, O, X, X, X, O],
        [O, O, O, O, O, O, X, X, X, O, O, O, O, O, O, X, X, X, X, X],
        [X, X, O, O, O, O, X, X, O, O, O, O, O, O, O, X, X, X, X, X],
        [X, X, X, O, X, X, X, O, O, O, O, O, X, X, X, X, X, X, X, O],
        [X, X, X, X, X, X, X, X, O, O, O, X, X, X, X, X, X, X, O, O],
        [O, O, O, X, X, X, X, X, X, O, O, X, X, X, X, X, X, O, O, O],
        [O, O, O, X, X, X, X, X, X, O, O, X, X, X, X, X, X, O, O, O],
        [O, O, O, O, X, X, X, X, X, O, O, O, X, X, X, X, X, O, O, O],
        [O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O],
        [O, O, O, O, O, O, O, X, X, X, X, X, X, X, O, O, O, O, O, O],
        [O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, O, O, O, O, O],
        [O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O],
        [O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O],
        [O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O],
        [O, O, O, O, O, O, X, X, X, X, X, X, X, O, O, O, O, O, O, O],
        [O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O],
        [O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O],
        [O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O],
        [O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O],
    ], dtype=int)
    soln = solve(rows, cols)
    assert_array_equal(soln, expected)


if __name__ == '__main__':
  unittest.main()
