#!/usr/bin/env python
from __future__ import print_function, division
import ast
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from webtool import webtool, webfn, webarg

from picross import solve, iter_solve, FULL, EMPTY, ConstraintsDetector


def main():
  ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  ap.add_argument('--port', type=int, default=8787, help='Port for web UI.')
  ap.add_argument('--image', help='Path to screenshot of puzzle.')
  ap.add_argument('--ocr-model-path', default='ocr.pkl',
                  help='Path to OCR model file.')
  ap.add_argument('--animate', action='store_true',
                  help='Use Bayesian solver to produce an animation.')
  ap.add_argument('row', nargs='?', help='Row constraints, comma-separated.')
  ap.add_argument('col', nargs='?', help='Column constraints, comma-separated.')
  args = ap.parse_args()

  if args.image:
    ocr = ConstraintsDetector(args.ocr_model_path)
    rows, cols = ocr.detect_constraints(args.image)
    print('Rows:', rows)
    print('Cols:', cols)
    ocr.save_model()
    return cli_main(rows, cols, pre_parsed=True, animate=args.animate)
  elif args.row is None and args.col is None:
    if args.animate:
      ap.error('Cannot pass --animate to web UI.')
    web_main.ocr_model_path = args.ocr_model_path
    return webtool(title='Picross', port=args.port, fn=web_main)
  if args.row is not None and args.col is not None:
    return cli_main(args.row, args.col, animate=args.animate)
  ap.error('Must supply both row and col constraints, or neither.')


def cli_main(rows, cols, pre_parsed=False, animate=False):
  if not pre_parsed:
    rows = parse_constraint(rows)
    cols = parse_constraint(cols)
  if animate:
    return animate_solution(rows, cols)
  soln = solve(rows, cols)
  out = np.full_like(soln, '_', dtype=bytes)
  out[soln == EMPTY] = ' '
  out[soln == FULL] = 'x'
  for row in out:
    print(''.join(row))


def animate_solution(rows, cols, anim_secs=5, end_secs=1):
  frames = list(iter_solve(rows, cols))
  print('Solved in', len(frames), 'steps.')

  fig, ax = plt.subplots(figsize=(6, 6))
  img = ax.imshow(frames[0], vmin=0, vmax=1)

  def _update(frame):
    img.set_data(frame)
    return (img,)

  interval = anim_secs * 1000 / len(frames)
  anim = FuncAnimation(fig, _update, frames=frames, blit=True, repeat=True,
                       interval=interval, repeat_delay=end_secs * 1000)
  anim
  plt.show()


@webfn('Solve', 'Provide either row/col constraints or an image file.',
       rows=webarg('Row constraints.', default='3,3,(3,1),5,(1,1,1)'),
       cols=webarg('Column constraints.', default='3,2,5,(2,1),5'),
       image=webarg('Puzzle screenshot.', type=open))
def web_main(state, rows='', cols='', image=None):
  if image is None:
    rows = parse_constraint(rows)
    cols = parse_constraint(cols)
  else:
    ocr = ConstraintsDetector(web_main.ocr_model_path)
    rows, cols = ocr.detect_constraints(image)
    ocr.save_model()
  soln = solve(rows, cols)
  width_px = 500 // soln.shape[1] - 1
  height_px = 500 // soln.shape[0] - 1
  html = [
      '<style type="text/css">',
      'table { border-collapse: collapse; }',
      'td { border: 1px solid darkgray;',
      'width: %dpx; height: %dpx; }' % (width_px, height_px),
      '.full { background-color: darkred; }',
      '.empty { background-color: khaki; }',
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
