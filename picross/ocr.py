from __future__ import print_function, division
import cv2
import numpy as np
import scipy.ndimage
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB


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
    col_nums, row_nums = find_row_col_numbers(path_or_file)

    # segment rows/cols of constraints
    row_labels, _ = scipy.ndimage.label(row_nums.any(axis=1))
    flat_col = col_nums.any(axis=0)
    scipy.ndimage.binary_closing(flat_col, iterations=4, output=flat_col)
    col_labels, _ = scipy.ndimage.label(flat_col)

    # parse constraints
    row_constraints = [self._parse_numbers(row_nums[s[0],:], horiz=True)
                       for s in scipy.ndimage.find_objects(row_labels)]
    col_constraints = [self._parse_numbers(col_nums[:,s[0]], horiz=False)
                       for s in scipy.ndimage.find_objects(col_labels)]
    return row_constraints, col_constraints

  def _parse_numbers(self, img, horiz=True):
    digit_labels, _ = scipy.ndimage.label(img)
    numbers = []
    in_2digit = False
    for s1, s2 in scipy.ndimage.find_objects(digit_labels):
      crop = img[s1,s2]
      if crop.size < 30 or min(crop.shape) < 4:
        continue
      if horiz:
        p1, p2 = s2.start, s1.start
      else:
        p1, p2 = s1.start, s2.start
      num_digits = crop.max()
      assert num_digits in (1, 2)  # sanity check
      if in_2digit:
        if not horiz:
          p1 = numbers[-1][0]
        in_2digit = False
      elif num_digits == 2:
        in_2digit = True
      digit = self.digit_rec.recognize(crop)
      numbers.append((p1, p2, num_digits, digit))
    if in_2digit:
      raise Exception('Incomplete 2-digit number: %d' % numbers[-1][-1])
    numbers.sort()

    tmp = []
    for _, __, num_digits, digit in numbers:
      if num_digits == 1:
        in_2digit = False
        tmp.append([digit])
      elif in_2digit:
        in_2digit = False
        tmp[-1].append(digit)
      else:
        in_2digit = True
        tmp.append([digit])
    assert not in_2digit
    return list(map(_digits2num, tmp))


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
    odds = proba[num] / max(1e-12, 1 - proba[num])
    if odds < 0.99 or not np.isfinite(odds):
      print(proba)
      ent = scipy.stats.entropy(proba)
      print("Unknown digit: %.1f%% sure it's a %d (ent=%g)" % (
            100*odds, num, ent))
      _print_digit(img)
      num = int(raw_input("What is it? "))
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
    clf = GaussianNB().partial_fit(X, y, classes=np.arange(10))
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
  edges = cv2.dilate(edges, None)

  # find puzzle area: axis-aligned big square
  contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
  squares = []
  for cnt in contours:
    # check contour size
    cnt_area = cv2.contourArea(cnt)
    if cnt_area < 10000:
      continue
    # check if contour area matches bbox area
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = w * h
    if not (0.95 < cnt_area / bbox_area < 1.05):
      continue
    # check squareness
    if abs(w - h)/w > 0.01:
      continue
    squares.append((w, h, x, y))

  if not squares:
    if debug:
      cv2.imwrite('out.png', edges)
      raise Exception('No squares found in edge image: see out.png')
    raise Exception('No squares found in edge image')

  w, h, x, y = max(squares)
  if min(w, h) < min(edges.shape) / 2:
    if debug:
      for w, h, x, y in squares:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
      cv2.imwrite('out.png', img)
      raise Exception('No big square found in edge image: see out.png')
    raise Exception('No big square found in edge image')

  col_nums = img[:y+1,x:x+w]
  row_nums = img[y:y+h,:x+1]

  # cv2.imwrite('col_img.png', col_nums)
  # cv2.imwrite('row_img.png', row_nums)
  col_nums = _label_pixels(col_nums)
  row_nums = _label_pixels(row_nums)
  # cv2.imwrite('col_lbl.png', col_nums * 127)
  # cv2.imwrite('row_lbl.png', row_nums * 127)
  return col_nums, row_nums


def _label_pixels(img):
  hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
  sat = hsv[:,:,1]
  if not 0.3 < sat.mean() < 0.7:
    raise Exception('Unexpected saturation distribution')

  # 1 -> single digit, 2 -> double digit, 0 -> background
  lbl = np.zeros_like(sat, dtype=np.uint8)
  lbl[sat < 0.3] = 1
  lbl[sat > 0.7] = 2
  return lbl
