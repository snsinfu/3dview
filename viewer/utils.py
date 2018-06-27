import math

import numpy as np


def as_float_array(ary):
    return np.array(ary, dtype=np.float32, copy=False)


def as_index_array(ary):
    return np.array(ary, dtype=np.uint32, copy=False)


def length(vec):
    return np.sqrt((np.array(vec)**2).sum(axis=-1))


def normalize(vec):
    return vec / (length(vec) + 1e-6)


def hsv_to_rgb(hue, satuation, value):
    phase = math.fmod(hue * 6, 6)
    sub_phase = math.fmod(phase, 2)
    x = value * (1 - satuation * max(0, sub_phase - 1))
    y = value * (1 - satuation * max(0, 1 - sub_phase))
    z = value * (1 - satuation)
    color = (x, y, z) if phase < 2 else \
            (z, x, y) if phase < 4 else \
            (y, z, x)
    return np.array(color)
