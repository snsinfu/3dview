import argparse
import itertools
import json
import sys

from PIL import Image
from PyQt4 import QtGui
import vispy.color

from colormaps import AVAILABLE_COLOR_MODULATORS, MatplotlibColorMap
from utils import as_float_array, length
from meshes import make_sphere_mesh
from modellers import AVAILABLE_MODELLERS
from scenes import TranslucentScene
from shaders import WireframeShader
from widgets import BufferedCanvas


DEFAULT_ANNOTATION_VALUE = 1.0
WALL_SPHERE_NDIV = 4


def main():
    app = QtGui.QApplication(sys.argv)
    args = parse_args()
    scene, scene_radius = create_scene(args)
    window = SceneDisplayWindow(scene, fov=(scene_radius * 1.2))
    window.show()
    sys.exit(app.exec_())


def parse_args():
    parser = argparse.ArgumentParser(description='.')

    # Option arguments
    parser.add_argument('-b', dest='bgcolor',
                        type=str, default='black',
                        help='Background color (default: black)')

    parser.add_argument('-w', dest='wall_color',
                        type=str, default='white,0.2',
                        help='Color and opacity of the wall (default: white,0.2)')

    parser.add_argument('-R', dest='wall_radius',
                        type=float, default=0,
                        help='Radius of the wall')

    # Positional arguments
    parser.add_argument('scheme',
                        type=argparse.FileType('r'),
                        help='3D modelling scheme file')

    parser.add_argument('coords', nargs='?',
                        type=argparse.FileType('r'), default=sys.stdin,
                        help='CXYZV input file (default: stdin)')

    return parser.parse_args()


class SceneDisplayWindow(QtGui.QWidget):
    def __init__(self, scene, fov=1.0):
        QtGui.QWidget.__init__(self, None)

        # Canvas to which the 3D model is rendered
        self._canvas = BufferedCanvas(scene, parent=self, rview=fov)

        # Save-to-file button
        savetofile = QtGui.QPushButton("Save", parent=self)
        savetofile.clicked.connect(self.save_to_file)

        # Layout
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self._canvas.native, 1)
        layout.addWidget(savetofile, 1)
        self.setLayout(layout)

    def save_to_file(self):
        extensions = ['png', 'bmp', 'gif', 'jpg']
        filter = 'Image (' + ' '.join('*.{}'.format(ext) for ext in extensions) + ')'
        filename = str(QtGui.QFileDialog.getSaveFileName(self, "Save Image", filter=filter))
        if filename:
            raw_image = self._canvas.framebuffer.read(alpha=False)
            img = Image.fromarray(raw_image)
            img.save(filename)


def create_scene(args):
    modellers = load_chain_modellers(args.scheme)
    center = (0, 0, 0)
    scene_radius = 0.0
    shaders = []

    records = parse_input(args.coords)

    def extract_group_tag(record):
        return record[0]

    for tag, subrecords in itertools.groupby(records, key=extract_group_tag):
        # Interpret a part of the tag as the name of the modelling scheme to
        # use for this subrecords.
        modeller_name = tag.split(':')[0]
        if modeller_name not in modellers:
            raise Exception('unknown modelling scheme: ' + modeller_name)
        modeller = modellers[modeller_name]

        points = []
        values = []
        for _, x, y, z, value in subrecords:
            points.append((x, y, z))
            values.append(value)
        points = as_float_array(points)
        shaders.append(modeller.make_shader(points, values))
        scene_radius = max(scene_radius, length(points - center).max())

    # Add wireframe spherical wall.
    if args.wall_radius > 0:
        color, opacity = args.wall_color.split(',', 1)
        color = vispy.color.Color(color).rgb
        opacity = float(opacity)

        wall = make_sphere_mesh(ndiv=WALL_SPHERE_NDIV)
        wall.vertices *= args.wall_radius
        wall.colors[:] = color
        shaders.append(WireframeShader(wall, opacity))

        scene_radius = max(scene_radius, args.wall_radius)

    return TranslucentScene(shaders, args.bgcolor), scene_radius


def parse_input(lines):
    """ Lazily parse input file. Each line of input file must consist of five
    space-separated fields that represent a single point with annotation
    value. The meaning of each field is:
      1. name of the chain the point belongs to
      2. x coordinate value of the point
      3. y coordinate value of the point
      4. z coordinate value of the point
      5. annotation value for the point (default: 1.0)
    """
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        record = line.split(None, 4)
        if len(record) == 4:
            record.append(DEFAULT_ANNOTATION_VALUE)

        tag, x, y, z, value = record

        yield tag, float(x), float(y), float(z), float(value)


def load_chain_modellers(inp):
    config = json.load(inp)
    modellers = dict()

    for chain_id, chain_config in config.iteritems():
        modeller = AVAILABLE_MODELLERS[chain_config['model']]
        base_colormap = MatplotlibColorMap(chain_config['color'], chain_config['color_range'])
        modulator = AVAILABLE_COLOR_MODULATORS[chain_config['modulator']]
        colormap = modulator(base_colormap, chain_config)
        modellers[chain_id] = modeller(colormap, chain_config)

    return modellers


if __name__ == '__main__':
    main()
