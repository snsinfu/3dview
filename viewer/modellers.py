# Modellers
#
# Modeller object creates shader object that renders given sequence of points
# in a defined way.

from colormaps import map_sequence_colors
from meshes import combine_meshes, generate_connective_cylinders, generate_spheres
from shaders import PolylineShader, MeshShader


# Number of mesh subdivision. Larger number = better quality.
SPHERE_NDIV = 3
CYLINDER_NDIV = 9


class PolylineModeller(object):
    """ Modeller for rendering sequence of points as polyline.
    """
    def __init__(self, cmap, config):
        self.cmap = cmap

    def make_shader(self, points, values):
        return PolylineShader(points, map_sequence_colors(values, self.cmap))


class SpheresModeller(object):
    """ Modeller for rendering points as spheres.
    """
    def __init__(self, cmap, config):
        self.cmap = cmap
        self.radius = config['radius']
        self.shininess = MeshShader.DEFAULT_SHININESS
        if 'shininess' in config:
            self.shininess = config['shininess']

    def make_shader(self, points, values):
        beads = generate_spheres(points, self.radius, ndiv=SPHERE_NDIV)
        colors = map_sequence_colors(values, self.cmap)
        for bead, color in zip(beads, colors):
            bead.colors[:] = color
        shader = MeshShader(combine_meshes(beads))
        shader.set_shininess(self.shininess)
        return shader


class LicoriceModeller(object):
    """ Modeller for rendering sequence of points as spheres connected by
    cylinders.
    """
    def __init__(self, cmap, config):
        self.cmap = cmap
        self.radius = config['radius']
        self.shininess = MeshShader.DEFAULT_SHININESS
        if 'shininess' in config:
            self.shininess = config['shininess']

    def make_shader(self, points, values):
        beads = generate_spheres(points, self.radius, ndiv=SPHERE_NDIV)
        bonds = generate_connective_cylinders(points, self.radius, ndiv=CYLINDER_NDIV)
        colors = map_sequence_colors(values, self.cmap)
        for bead, color in zip(beads, colors):
            bead.colors[:] = color
        for i, bond in enumerate(bonds):
            bond.colors[0::2] = colors[i]
            bond.colors[1::2] = colors[i + 1]
        shader = MeshShader(combine_meshes(beads + bonds))
        shader.set_shininess(self.shininess)
        return shader


AVAILABLE_MODELLERS = {
    'polyline': PolylineModeller,
    'beads':    SpheresModeller,
    'licorice': LicoriceModeller
}
