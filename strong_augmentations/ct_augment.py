from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np

# the maximum number of parameters for a single augmentation function. The "rescale" function is the only one which has
# 2 parameters: the scale and the method of rescaling.
MAX_PARAMS = 2


def _enhance(x, op, level):
    return op(x).enhance(0.1 + 1.9 * level)


def _imageop(x, op, level):
    return Image.blend(x, op(x), level)


def _filter(x, op, level):
    return Image.blend(x, x.filter(op), level)


def autocontrast(x, level):
    return _imageop(x, ImageOps.autocontrast, level)


def blur(x, level):
    return _filter(x, ImageFilter.BLUR, level)


def brightness(x, brightness):
    return _enhance(x, ImageEnhance.Brightness, brightness)


def color(x, color):
    return _enhance(x, ImageEnhance.Color, color)


def contrast(x, contrast):
    return _enhance(x, ImageEnhance.Contrast, contrast)


def cutout(x, level):
    """Apply cutout to pil_img at the specified level."""
    size = 1 + int(level * min(x.size) * 0.499)
    img_height, img_width = x.size
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2), min(img_width, width_loc + size // 2))
    pixels = x.load()  # create the pixel map
    for i in range(upper_coord[0], lower_coord[0]):  # for every col:
        for j in range(upper_coord[1], lower_coord[1]):  # For every row
            pixels[i, j] = (127, 127, 127)  # set the color accordingly
    return x


def equalize(x, level):
    return _imageop(x, ImageOps.equalize, level)


def invert(x, level):
    return _imageop(x, ImageOps.invert, level)


def identity(x):
    return x


def posterize(x, level):
    level = 1 + int(level * 7.999)
    return ImageOps.posterize(x, level)


def rescale(x, scale, method):
    s = x.size
    scale *= 0.25
    crop = (scale * s[0], scale * s[1], s[0] * (1 - scale), s[1] * (1 - scale))
    methods = (Image.ANTIALIAS, Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.NEAREST)
    method = methods[int(method * 5.99)]
    return x.crop(crop).resize(x.size, method)


def rotate(x, angle):
    angle = int(np.round((2 * angle - 1) * 45))
    return x.rotate(angle)


def sharpness(x, sharpness):
    return _enhance(x, ImageEnhance.Sharpness, sharpness)


def shear_x(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


def shear_y(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


def smooth(x, level):
    return _filter(x, ImageFilter.SMOOTH, level)


def solarize(x, th):
    th = int(th * 255.999)
    return ImageOps.solarize(x, th)


def translate_x(x, delta):
    delta = (2 * delta - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))


def translate_y(x, delta):
    delta = (2 * delta - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))


def fixmatch_cta_pool():
    # FixMatch paper
    augs = [(autocontrast, [17]),
            (blur, [17]),
            (brightness, [17]),
            (color, [17]),
            (contrast, [17]),
            (cutout, [17]),
            (equalize, [17]),
            (identity, []),
            (invert, [17]),
            (posterize, [8]),
            (rescale, [17, 6]),
            (rotate, [17]),
            (sharpness, [17]),
            (shear_x, [17]),
            (shear_y, [17]),
            (solarize, [17]),
            (smooth, [17]),
            (translate_x, [17]),
            (translate_y, [17])]
    return augs


class CTAugment:
    """
    This class is adapted from the original FixMatch algorithm, and used in the paper as the strong augmentation
    in the semi-supervised phase.
    """
    def __init__(self, n=2):
        self.n = n
        self.augment_pool = fixmatch_cta_pool()
        self.rates = {f.__name__: [np.ones(param_bins) for param_bins in bins] for f, bins in self.augment_pool}
        self.decay = 0.99
        self.confidence_threshold = 0.8

    def rate_to_p(self, rate):
        p = rate + (1 - self.decay)  # Avoid to have all zero.
        p = p / p.max()
        p[p < self.confidence_threshold] = 0
        return p / p.sum()

    def __call__(self, img, probe=False):
        ops_indices = np.random.choice(len(self.augment_pool), size=self.n)
        ops = [self.augment_pool[index] for index in ops_indices]
        mag = np.full((self.n, MAX_PARAMS), -1, dtype=np.float32)
        for i, (index, (op, bins)) in enumerate(zip(ops_indices, ops)):
            if probe:  # labeled images for updating the rates.
                v = np.random.uniform(0, 1, len(bins)).tolist()
                mag[i, :len(bins)] = v
            else:  # unlabeled images, we sample from the learned rates.
                v = []
                for j, param_bins in enumerate(bins):
                    rnd = np.random.uniform(0, 1)
                    p = self.rate_to_p(self.rates[op.__name__][j])
                    mag_choice = np.random.choice(param_bins, p=p)
                    v.append((mag_choice + rnd) / param_bins)
                mag[i, :len(v)] = v

            img = op(img, *v)
        if probe:
            return img, ops_indices, mag
        return cutout(img, 1)

    def update(self, proximity, ops_indices_update, mags_update, print_probs=False):
        alpha = 1 - self.decay
        for i, (op_indices, mags) in enumerate(zip(ops_indices_update, mags_update)):
            for op_idx, param_mags in zip(op_indices, mags):
                for j, mag in enumerate(param_mags):
                    if mag == -1:
                        break
                    op = self.augment_pool[op_idx]
                    op_name, n_bins = op[0].__name__, op[1][j]
                    mag_index = int(mag * n_bins * 0.999)
                    self.rates[op_name][j][mag_index] += (proximity[i] - self.rates[op_name][j][mag_index]) * alpha
        if print_probs:
            print('\n'.join('%-16s    %s' % (k, ' / '.join(' '.join('%.2f' % x for x in self.rate_to_p(rate))
                  for rate in self.rates[k]))
                  for k in sorted(self.rates.keys())))