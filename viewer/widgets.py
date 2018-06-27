import math

import numpy as np
from PIL import Image
from PyQt4 import QtCore, QtGui
from vispy import gloo
from vispy.app import Canvas
from vispy.util import transforms
from vispy.util.quaternion import Quaternion

from shaders import TextureDisplayShader


class InteractiveViewportMixin(object):
    """ Canvas mixin for interactive rotation, translation and zooming

    This class implements on_resize, on_mouse_move, on_mouse_press,
    on_mouse_release and on_mouse_wheel in such a way that the view matrix and
    the projection matrix changes in response to user input.

    The derived class should implement on_update_view and on_update_projection
    to update the matrices in the shader.
    """

    def __init__(self, rview, zview=None, zoom_speed=0.1, move_speed=0.01):
        """ Constructor

        Parameters
        ----------
        rview : float
            Radius of inscribed sphere within the field of view.
        zview : float
            Distance to the z-clip plane from the origin.
        zoom_speed : float
            How sensitive the field of view to mouse wheel rolling.
        move_speed : float
            How sensitive the translation of view to mouse dragging.
        """
        self.rview = rview
        self.zview = zview if zview else rview
        self.zoom_speed = zoom_speed
        self.move_speed = move_speed
        self._translation = (0.0, 0.0, 0.0)
        self._quaternion = Quaternion()

    def on_update_view(self, view):
        """ Handle view matrix change
        """
        pass

    def on_update_projection(self, proj):
        """ Handle projection matrix change
        """
        pass

    def get_view_matrix(self):
        """ Return current view matrix
        """
        return np.dot(self._quaternion.get_matrix(),
                      transforms.translate(self._translation))

    def get_projection_matrix(self):
        """ Return current projection matrix
        """
        # Determine the field of view of orthographic projection so that the
        # sphere of radius rview centered at the origin inscribes the field
        # of view.
        width, height = self.physical_size
        aspect = float(width) / height
        if aspect > 1:
            top = self.rview
            right = self.rview * aspect
        else:
            top = self.rview / aspect
            right = self.rview
        assert min(top, right) == self.rview
        left = -right
        bottom = -top
        near = self.zview
        far = -self.zview

        left -= 1e-6
        right += 1e-6
        bottom -= 1e-6
        top += 1e-6
        near -= 1e-6
        far += 1e-6

        return transforms.ortho(left, right, bottom, top, near, far)

    def _update_view(self):
        self.on_update_view(self.get_view_matrix())

    def _update_projection(self):
        self.on_update_projection(self.get_projection_matrix())

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)
        self._update_projection()

    def on_mouse_move(self, event):
        """ Left drag to rotate, right drag to tranelate
        """
        if event.is_dragging and event.last_event:
            x0, y0 = event.last_event.pos
            x1, y1 = event.pos
            w, h = self.size

            if event.button == 1:
                self._quaternion *= arcball(x0, y0, w, h) * arcball(x1, y1, w, h)
                self._update_view()

            if event.button == 2:
                tx, ty, tz = self._translation
                tx += self.move_speed * (x1 - x0)
                ty += -self.move_speed * (y1 - y0)
                self._translation = (tx, ty, tz)
                self._update_view()

    def on_mouse_wheel(self, event):
        """ Roll mouse wheel to zoom in/out
        """
        roll = event.delta[1]
        self.rview *= math.exp(-self.zoom_speed * roll)
        self._update_projection()

    def on_mouse_press(self, event):
        if event.button == 1:
            cursor = QtGui.QCursor(QtCore.Qt.ClosedHandCursor)
            QtGui.QApplication.setOverrideCursor(cursor)

        if event.button == 2:
            cursor = QtGui.QCursor(QtCore.Qt.SizeAllCursor)
            QtGui.QApplication.setOverrideCursor(cursor)

    def on_mouse_release(self, event):
        QtGui.QApplication.restoreOverrideCursor()


def arcball(x, y, w, h):
    r = (w + h) / 2.0
    x = -(2.0 * x - w) / r
    y =  (2.0 * y - h) / r
    h = math.hypot(x, y)
    if h > 1:
        return Quaternion(0, x/h, y/h, 0)
    else:
        return Quaternion(0, x, y, math.sqrt(1 - h**2))


class BufferedCanvas(InteractiveViewportMixin, Canvas):
    """ Interactive canvas with FBO support
    """

    def __init__(self, program,
                 rview=1, zview=None, zoom_speed=0.1, move_speed=0.01,
                 *args, **kwargs):
        """ Constructor
        """
        InteractiveViewportMixin.__init__(self, rview, zview, zoom_speed, move_speed)
        Canvas.__init__(self, *args, **kwargs)

        #
        self._program = program
        self._program.set_projection_matrix(self.get_projection_matrix())
        self._program.set_view_matrix(self.get_view_matrix())

        # Set up framebuffer
        width, height = self.physical_size
        shape = (height, width, 3)
        rendertex = gloo.Texture2D(shape)
        self._fbo = gloo.FrameBuffer(rendertex, gloo.RenderBuffer(shape))
        self._display_program = TextureDisplayShader(rendertex)

    @property
    def framebuffer(self):
        """ Return the framebuffer object
        """
        return self._fbo

    def on_update_view(self, view):
        self._program.set_view_matrix(view)
        self.update()

    def on_update_projection(self, proj):
        self._program.set_projection_matrix(proj)
        self.update()

    def on_resize(self, event):
        width, height = event.physical_size
        self._fbo.resize((height, width))
        gloo.set_viewport(0, 0, width, height)
        InteractiveViewportMixin.on_resize(self, event)

    def on_draw(self, event):
        with self._fbo:
            gloo.set_viewport(0, 0, *self.physical_size)
            self._program.draw()
        self._display_program.draw()
