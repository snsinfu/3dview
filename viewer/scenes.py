from vispy import gloo

from shaders import CompositeShader


class TranslucentScene(object):
    def __init__(self, shaders, background='black'):
        self._shader = CompositeShader(shaders)
        self._background = background

    def set_view_matrix(self, view):
        self._shader.set_view_matrix(view)

    def set_projection_matrix(self, proj):
        self._shader.set_projection_matrix(proj)

    def draw(self):
        gloo.set_state('translucent')
        gloo.clear(self._background, depth=True)
        gloo.gl.glClear(gloo.gl.GL_DEPTH_BUFFER_BIT) # XXX vispy does not clear depth buffer
        self._shader.draw()
