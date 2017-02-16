#!/usr/bin/env python
import ast
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from webtool import webtool, webfn, webarg

from picross import solve, FULL, EMPTY


def main():
  ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  ap.add_argument('--port', type=int, default=8787, help='Port for web UI.')
  ap.add_argument('row', nargs='?', help='Row constraints, comma-separated.')
  ap.add_argument('col', nargs='?', help='Column constraints, comma-separated.')
  args = ap.parse_args()

  if args.row is None and args.col is None:
    return webtool(title='Picross', port=args.port, fn=web_main)
  if args.row is not None and args.col is not None:
    return cli_main(args.row, args.col)
  ap.error('Must supply both row and col constraints, or neither.')


def cli_main(rows, cols):
  rows = parse_constraint(rows)
  cols = parse_constraint(cols)
  soln = solve(rows, cols)
  out = np.full_like(soln, ' ', dtype=bytes)
  out[soln == EMPTY] = '_'
  out[soln == FULL] = 'x'
  for row in out:
    print(''.join(row))


@webfn('Solve', '',
       rows=webarg('Row constraints.', default='3,3,(3,1),5,(1,1,1)'),
       cols=webarg('Column constraints.', default='3,2,5,(2,1),5'))
def web_main(state, rows, cols):
  rows = parse_constraint(rows)
  cols = parse_constraint(cols)
  soln = solve(rows, cols)
  width_px = 500 // soln.shape[1]
  height_px = 500 // soln.shape[0]
  html = [
      '<style type="text/css">',
      'table { border-collapse: collapse; }',
      'td { width: %dpx; height: %dpx; }' % (width_px, height_px),
      '.full { background-color: black; }',
      '.empty { background-color: gray; }',
      '</style>'
      '<table>'
  ]
  for row in soln:
    html.append('<tr>')
    for x in row:
      if x == FULL:
        html.append('<td class="full"></td>')
      elif x == EMPTY:
        html.append('<td class="empty"></td>')
      else:
        html.append('<td></td>')
    html.append('</tr>')
  html.append('</table>')
  return '\n'.join(html)


def parse_constraint(constraint_string):
  c = ast.literal_eval('[%s]' % constraint_string)
  return [x if hasattr(x, '__len__') else [x] for x in c]


if __name__ == '__main__':
  main()
