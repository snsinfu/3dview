# Sequential Color Maps
#
# Sequential color map object calculates color of chain element using
# sequential position and feature value of each element.

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


def map_sequence_colors(values, cmap):
    """ Map feature values to colors using given sequential color map.
    """
    seq = np.linspace(0, 1, num=len(values))
    return [cmap.to_rgb(s, v) for s, v in zip(seq, values)]


class MatplotlibColorMap(object):
    """ Sequential color map that uses matplotlib color map to generate
    color for given sequential position. Feature values are not used.
    """
    def __init__(self, cmap, minmax=(0, 1)):
        norm = Normalize(vmin=0, vmax=1)
        self._mapper = ScalarMappable(norm, cmap)
        self._minmax = minmax

    def to_rgb(self, seq, feat):
        val = self._minmax[0] * (1 - seq) + self._minmax[1] * seq
        return self._mapper.to_rgba(val)[:3]


class NonModulatedColorMap(object):
    """ Sequential color map wrapper that just forwards calculation to the
    underlying sequential color map.
    """
    def __init__(self, base, config):
        self._base = base

    def to_rgb(self, seq, feat):
        return self._base.to_rgb(seq, feat)


class SatuationModulatedColorMap(object):
    """ Sequential color map wrapper that modulates satuation component of the
    color calculated by the underlying sequential color map by feature value.
    """
    def __init__(self, base, config):
        self._base = base

    def to_rgb(self, seq, feat):
        base_color = self._base.to_rgb(seq, feat)
        gray = max(base_color)
        return tuple(feat * x + (1 - feat) * gray for x in base_color)


class BrightnessModulatedColorMap(object):
    """ Sequential color map wrapper that modulates brightness component of the
    color calculated by the underlying sequential color map by feature value.
    """
    def __init__(self, base, config):
        self._base = base

    def to_rgb(self, seq, feat):
        base_color = self._base.to_rgb(seq, feat)
        return tuple(feat * x for x in color)


class ColorMapModulatedColorMap(object):
    """ Sequential color map wrapper that passes feature value to the underlying
    color map as sequential position.
    """
    def __init__(self, base, config):
        self._base = base

    def to_rgb(self, seq, feat):
        return self._base.to_rgb(feat, feat)


AVAILABLE_COLOR_MODULATORS = {
    'none':       NonModulatedColorMap,
    'satuation':  SatuationModulatedColorMap,
    'brightness': BrightnessModulatedColorMap,
    'colormap':   ColorMapModulatedColorMap
}
