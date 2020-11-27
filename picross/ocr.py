from __future__ import print_function, division
import cv2
import joblib
import numpy as np
import scipy.ndimage
from sklearn.naive_bayes import MultinomialNB


class ConstraintsDetector(object):
  def __init__(self, model_path):
    self.model_path = model_path
    try:
      self.digit_rec = NaiveBayesDigits.load(self.model_path)
    except:
      self.digit_rec = ExactLookupDigits()

  def save_model(self):
    if isinstance(self.digit_rec, ExactLookupDigits):
      digits, imgs = zip(*self.digit_rec.known_pairs)
      self.digit_rec = NaiveBayesDigits.train(imgs, digits)
    self.digit_rec.save(self.model_path)

  def detect_constraints(self, path_or_file):
    # crop to constraint areas
    col_nums, row_nums = find_row_col_numbers(path_or_file, debug=False)

    # segment rows/cols of constraints
    row_labels, _ = scipy.ndimage.label(row_nums.any(axis=1))
    flat_col = col_nums.any(axis=0)
    scipy.ndimage.binary_closing(flat_col, iterations=4, output=flat_col)
    col_labels, _ = scipy.ndimage.label(flat_col)

    # parse constraints
    row_constraints = []
    for s in scipy.ndimage.find_objects(row_labels):
      nums = self._parse_numbers(row_nums[s[0],:], horiz=True)
      if nums:
        row_constraints.append(nums)
    col_constraints = []
    for s in scipy.ndimage.find_objects(col_labels):
      nums = self._parse_numbers(col_nums[:,s[0]], horiz=False)
      if nums:
        col_constraints.append(nums)
    return row_constraints, col_constraints

  def _parse_numbers(self, img, horiz=True):
    digit_labels, _ = scipy.ndimage.label(img)
    digit_locs = scipy.ndimage.find_objects(digit_labels)
    # sort the digit locations
    if horiz:
      # by column (left to right)
      digit_locs.sort(key=lambda loc: loc[1].start)
    else:
      # by row (top to bottom)
      digit_locs.sort(key=lambda loc: loc[0].start)

    numbers = []
    tmp_2digit = []
    for loc in digit_locs:
      crop = img[loc]
      if crop.size < 30 or min(crop.shape) < 4:
        continue
      num_digits = crop.max()
      assert num_digits in (1, 2)  # sanity check
      digit = self.digit_rec.recognize(crop)
      if num_digits == 1:
        assert not tmp_2digit, 'Incomplete 2-digit number: %s' % numbers
        numbers.append([digit])
      else:
        tmp_2digit.append((loc[1].start, digit))
        if len(tmp_2digit) == 2:
          numbers.append([d for _, d in sorted(tmp_2digit)])
          tmp_2digit = []

    assert not tmp_2digit, 'Incomplete 2-digit number: %s' % numbers
    return list(map(_digits2num, numbers))


class DigitRecognizer(object):
  digit_shape = (8, 6)

  def recognize(self, img):
    # prep the image
    img = (img > 0).astype(float)
    nr, nc = self.digit_shape
    scale = (nr/img.shape[0], nc/img.shape[1])
    img = scipy.ndimage.zoom(img, scale, order=1)
    np.clip(img, 0, 1, out=img)

    # predict
    proba = self._predict_proba(img)
    num = np.argmax(proba)

    # check that we have reasonable confidence
    ent = scipy.stats.entropy(proba)
    if ent > 0.02 or not np.isfinite(ent):
      print(proba)
      print("Unknown digit: I think it's a %d (ent=%g)" % (num, ent))
      _print_digit(img)
      num_str = input("What is it? ")
      num = int(num_str) if num_str else num
      self._update_model(img, num)

    return num


class ExactLookupDigits(DigitRecognizer):
  def __init__(self):
    self.known_pairs = []

  def _predict_proba(self, img):
    proba = np.zeros(10)
    for digit, known_img in self.known_pairs:
      if np.allclose(img, known_img):
        proba[digit] = 1
        break
    else:
      proba[:] = 0.1
    return proba

  def _update_model(self, img, digit):
    self.known_pairs.append((digit, img))


class NaiveBayesDigits(DigitRecognizer):
  def __init__(self, clf, dirty=True):
    self.clf = clf
    self.dirty = dirty

  @staticmethod
  def train(known_images, known_digits):
    y = np.array(known_digits)
    X = np.array(known_images).reshape((len(y), -1))
    clf = MultinomialNB().partial_fit(X, y, classes=np.arange(10))
    return NaiveBayesDigits(clf)

  @staticmethod
  def load(path):
    clf = joblib.load(path)
    return NaiveBayesDigits(clf, dirty=False)

  def save(self, path):
    if self.dirty:
      joblib.dump(self.clf, path)
      self.dirty = False

  def _predict_proba(self, img):
    return self.clf.predict_proba(img.reshape((1, -1)))[0]

  def _update_model(self, img, digit):
    self.clf.partial_fit(img.reshape((1, -1)), [digit])
    self.dirty = True


class RunningMeanDigits(DigitRecognizer):
  def __init__(self):
    r, c = self.digit_shape
    self.tpl_sums = np.zeros((10, r, c), dtype=float)
    self.tpl_counts = np.zeros(10, dtype=int)
    self.tpl_mean = self.tpl_sums.copy()

  def _predict_proba(self, img):
    dist = np.linalg.norm(self.tpl_mean - img, axis=(1, 2))
    proba = dist.max() - dist
    norm = proba.sum()
    if norm > 0:
      proba /= proba.sum()
    return proba

  def _update_model(self, img, digit):
    self.tpl_sums[digit] += img
    self.tpl_counts[digit] += 1
    self.tpl_mean[digit] = self.tpl_sums[digit] / self.tpl_counts[digit]


def _print_digit(arr):
  chars = np.array(list(' .:#'))
  thresh = np.array([0, 0.1, 0.5, 1])
  for row in chars[np.searchsorted(thresh, arr)]:
    print(''.join(row))


def _digits2num(digits):
  # hax, but whatever
  return int(''.join(map(str, digits)))


def find_row_col_numbers(fpath, debug=False):
  if hasattr(fpath, 'read'):
    buf = np.asarray(bytearray(fpath.read()), dtype=np.uint8)
    img = cv2.imdecode(buf, 1)
  else:
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

  # clean up edges via closing
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
  if debug:
    cv2.imwrite('00-edges.png', edges)

  # remove any non-axis-aligned edges
  h, w = edges.shape
  lines = cv2.HoughLines(edges, 1, np.pi / 4, int(min(h, w) * 0.75))
  rho, theta = lines[:, 0].T
  vert_xs = rho[theta < 0.1].astype(int)
  horiz_ys = rho[theta > 1.5].astype(int)
  mask = np.zeros_like(edges)
  for x in vert_xs:
      cv2.line(mask, (x, 0), (x, h), 255, 3)
  for y in horiz_ys:
      cv2.line(mask, (0, y), (w, y), 255, 3)
  edges = cv2.bitwise_and(edges, mask)
  if debug:
    cv2.imwrite('01-axis_edges.png', edges)

  # find puzzle area: axis-aligned big square
  padded = np.zeros((h + 2, w + 2), dtype=edges.dtype)
  padded[1:-1, 1:-1] = edges
  cv2.floodFill(padded, None, (0, 0), 255)
  square = cv2.bitwise_not(padded[1:-1, 1:-1])
  if debug:
    cv2.imwrite('02-square.png', square)

  # get the bounding box of the big square
  contours = cv2.findContours(square, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[0]
  hull = cv2.convexHull(np.vstack(contours))
  x, y, w, h = cv2.boundingRect(hull)

  col_nums = img[:y+1,x:x+w]
  row_nums = img[y:y+h,:x+1]

  if debug:
    cv2.imwrite('03-col_img.png', col_nums)
    cv2.imwrite('05-row_img.png', row_nums)
  col_nums = _label_pixels(col_nums, vertical_gradient=False)
  row_nums = _label_pixels(row_nums, vertical_gradient=True)
  if debug:
    cv2.imwrite('04-col_lbl.png', col_nums * 127)
    cv2.imwrite('06-row_lbl.png', row_nums * 127)
  return col_nums, row_nums


def _label_pixels(img, vertical_gradient=True):
  hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
  # look at the saturation channel
  sat = hsv[:,:,1]
  # quantize to 10 discrete bins
  qsat = np.digitize(sat, np.linspace(0, 1, 11)[1:])

  if vertical_gradient:
    # the background is the bin with the largest count in a given row
    bg_sat = np.array([np.bincount(row).argmax() for row in qsat])[:,None]
  else:
    # the background is the bin with the largest count
    bg_sat = np.bincount(qsat.ravel()).argmax()

  # 1 -> single digit, 2 -> double digit, 0 -> background
  lbl = np.zeros_like(sat, dtype=np.uint8)
  lbl[qsat < bg_sat] = 1
  lbl[qsat > bg_sat] = 2

  # do a little cleanup: there should be >10% background in each row and column
  bad_cols = np.count_nonzero(lbl, axis=0) > (lbl.shape[0] * 0.9)
  bad_rows = np.count_nonzero(lbl, axis=1) > (lbl.shape[1] * 0.9)
  lbl[:, bad_cols] = 0
  lbl[bad_rows, :] = 0
  return lbl
