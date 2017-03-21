from __future__ import print_function, division
import cv2
import numpy as np
import scipy.ndimage
from sklearn.metrics import pairwise_distances_argmin


class ConstraintsDetector(object):
  def __init__(self, tpl_path):
    self.tpl_path = tpl_path
    self.num_templates_on_disk = 0
    self.templates = []

  def load_templates(self):
    self.num_templates_on_disk = 0
    self.templates = []
    try:
      npz = np.load(self.tpl_path)
    except:
      return
    for key in npz.files:
      num = int(key.split(':', 1)[0])
      self.templates.append((num, npz[key]))
    self.num_templates_on_disk = len(self.templates)

  def save_templates(self):
    if self.num_templates_on_disk == len(self.templates):
      return
    if not self.tpl_path:
      return
    save_dict = {}
    for ct, (num, tpl) in enumerate(self.templates):
      key = '%d:%d' % (num, ct)
      save_dict[key] = tpl
    np.savez(self.tpl_path, **save_dict)
    self.num_templates_on_disk = len(self.templates)

  def detect_constraints(self, path_or_file):
    # crop to constraint areas
    col_nums, row_nums = find_row_col_numbers(path_or_file)

    # segment rows/cols of constraints
    row_labels, num_rows = scipy.ndimage.label(row_nums.any(axis=1))
    col_labels, num_cols = scipy.ndimage.label(col_nums.any(axis=0))

    # parse constraints
    row_constraints = [
        parse_numbers(row_nums[row_labels == x,:], self.templates, horiz=True)
        for x in range(1, num_rows+1)]
    col_constraints = [
        parse_numbers(col_nums[:,col_labels == x], self.templates, horiz=False)
        for x in range(1, num_cols+1)]
    return row_constraints, col_constraints

  def __enter__(self):
    self.load_templates()
    return self

  def __exit__(self, *args):
    self.save_templates()


def parse_numbers(img, templates, horiz=True):
  if img.max() == 0:
    return []
  digit_labels, num_digits = scipy.ndimage.label(img)
  numbers = []
  for s1, s2 in scipy.ndimage.find_objects(digit_labels):
    crop = img[s1,s2]
    pos = s2.start if horiz else s1.start
    val = crop.max()
    assert val in (1, 2)  # sanity check
    num = _match_number(crop > 0, templates)
    numbers.append((pos, val, num))
  numbers.sort()

  tmp = []
  in_2digit = False
  for _, val, num in numbers:
    if val == 1:
      in_2digit = False
      tmp.append([num])
    elif in_2digit:
      tmp[-1].append(num)
    else:
      in_2digit = True
      tmp.append([num])
  return list(map(_digits2num, tmp))


def _digits2num(digits):
  # hax, but whatever
  return int(''.join(map(str, digits)))


def _match_number(img, templates):
  for num, tpl in templates:
    if img.shape == tpl.shape and np.count_nonzero(img != tpl) <= 5:
      break
  else:
    print("Unknown number:")
    print(img.astype(int))
    num = int(raw_input("What is it? "))
    templates.append((num, img))
  return num


def find_row_col_numbers(fpath, debug=False):
  img = cv2.imread(fpath)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # invert and normalize
  gray = 255. - gray.astype(float)
  gray -= gray.min()
  gray /= gray.max()
  gray *= 255
  gray = gray.astype(np.uint8)

  # detect edges
  edges = cv2.Canny(gray, 50, 200, apertureSize=3)
  edges = cv2.dilate(edges, None)

  # find puzzle area: big square
  contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
  squares = []
  for cnt in contours:
    cnt_len = cv2.arcLength(cnt, True)
    cnt_area = cv2.contourArea(cnt)
    s1 = cnt_len / 4.
    s2 = np.sqrt(cnt_area)
    if cnt_area < 10000 or abs(s1 - s2)/s1 > 0.01:
      continue
    cnt = cv2.approxPolyDP(cnt, 0.01*cnt_len, True)
    if len(cnt) != 4 or not cv2.isContourConvex(cnt):
      continue
    cnt = cnt.reshape(-1, 2)
    squares.append((s1, cnt))
  if not squares:
    if debug:
      cv2.imwrite('out.png', edges)
      raise Exception('No squares found in edge image: see out.png')
    raise Exception('No squares found in edge image')
  edge_len, big_square = max(squares, key=lambda t: t[0])
  if edge_len < min(edges.shape) / 2:
    if debug:
      cv2.drawContours(img, squares, -1, (0, 255, 0), 3)
      cv2.imwrite('out.png', img)
      raise Exception('No big square found in edge image: see out.png')
    raise Exception('No big square found in edge image')

  # crop to just the numbers areas
  j1, i1 = big_square.min(axis=0)
  j2, i2 = big_square.max(axis=0) + 1
  col_nums = img[:i1+1,j1:j2]
  row_nums = img[i1:i2,:j1+1]

  # 1 -> single digit, 2 -> double digit, 0 -> background
  col_nums = _label_pixels(col_nums)
  row_nums = _label_pixels(row_nums)
  return col_nums, row_nums


def _label_pixels(img):
  # blue, white, yellow -> (0, 1, 2)
  target_hues = np.deg2rad(np.array([240, 0, 60]))

  hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
  hue = np.deg2rad(hsv[:,:,0].ravel())

  target_sincos = np.column_stack((np.sin(target_hues), np.cos(target_hues)))
  sincos = np.column_stack((np.sin(hue), np.cos(hue)))
  quantized = pairwise_distances_argmin(sincos, target_sincos)
  return quantized.reshape(img.shape[:2])


if __name__ == '__main__':
  main()
