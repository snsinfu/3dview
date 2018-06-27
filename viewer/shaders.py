import math

import numpy as np
from vispy import gloo

from utils import as_float_array


class PointShader(object):
    """ Shader program for rendering points in specified colors.
    """
    def __init__(self, points, colors=None, color=None):
        # Color is white by default.
        if colors is None:
            colors = np.empty((len(points), 3), dtype=np.float32)
            colors[:] = (1.0, 1.0, 1.0) if color is None else color

        self._shader = gloo.Program(self.VERT_SHADER, self.FRAG_SHADER)
        self._shader['position'] = as_float_array(points)
        self.set_colors(colors)

        I = np.eye(4, dtype=np.float32)
        self.set_view_matrix(I)
        self.set_projection_matrix(I)

    def set_colors(self, colors):
        self._shader['color'] = as_float_array(colors)

    def set_view_matrix(self, view):
        self._shader['u_view'] = as_float_array(view)

    def set_projection_matrix(self, proj):
        self._shader['u_projection'] = as_float_array(proj)

    def draw(self):
        self._shader.draw('points')


PointShader.VERT_SHADER = """
#version 120

uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec3 position;
attribute vec3 color;

varying vec3 v_color;

void main(void)
{
    gl_Position = u_projection * u_view * vec4(position, 1.0);
    v_color = color;
}
"""

PointShader.FRAG_SHADER = """
#version 120

varying vec3 v_color;

void main()
{
    gl_FragColor = vec4(v_color, 1.0);
}
"""


class SpriteShader(object):
    """ Shader program for rendering textured rectangles.
    """
    def __init__(self, points, texture, colors=None, color=None, sizes=None, size=None):
        # Color is white by default.
        if colors is None:
            colors = np.empty((len(points), 3), dtype=np.float32)
            colors[:] = (1, 1, 1) if color is None else color
        # Size is 1.0 by default.
        if sizes is None:
            sizes = np.empty((len(points), ), dtype=np.float32)
            sizes[:] = 1.0 if size is None else size

        self._shader = gloo.Program(self.VERT_SHADER, self.FRAG_SHADER)
        self._shader['position'] = as_float_array(points)
        self._shader['color'] = as_float_array(colors)
        self._shader['size'] = as_float_array(sizes)
        self._shader['u_texture'] = texture

        I = np.eye(4, dtype=np.float32)
        self.set_view_matrix(I)
        self.set_projection_matrix(I)

    def set_view_matrix(self, view):
        self._shader['u_view'] = as_float_array(view)

    def set_projection_matrix(self, proj):
        self._shader['u_projection'] = as_float_array(proj)

    def draw(self):
        self._shader.draw('points')

SpriteShader.VERT_SHADER = """
#version 120

uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec3 position;
attribute vec3 color;
attribute float size;

varying vec3 v_color;

void main(void)
{
    mat4 mvp = u_projection * u_view;
    float projected_size = length(mat3(u_projection) * vec3(size, 0, 0));
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = projected_size;
    v_color = color;
}
"""

SpriteShader.FRAG_SHADER = """
#version 120

uniform sampler2D u_texture;

varying vec3 v_color;

void main()
{
    vec4 tex_color = texture2D(u_texture, gl_PointCoord);
    if (tex_color.a < 0.1)
        discard;
    else
        gl_FragColor = tex_color * vec4(v_color, 1.0);
}
"""


class PolylineShader(object):
    """ Shader program for rendering polyline.
    """
    def __init__(self, points, colors=None, color=None, linewidth=1):
        # Color is white by default.
        if colors is None:
            colors = np.empty((len(points), 3), dtype=np.float32)
            colors[:] = color if color else (1, 1, 1)

        self._shader = gloo.Program(self.VERT_SHADER, self.FRAG_SHADER)
        self._shader['position'] = as_float_array(points)
        self._shader['color'] = as_float_array(colors)
        self._linewidth = linewidth

        I = np.eye(4, dtype=np.float32)
        self.set_model_matrix(I)
        self.set_view_matrix(I)
        self.set_projection_matrix(I)

    def set_model_matrix(self, mmat):
        self._shader['u_model'] = as_float_array(mmat)

    def set_view_matrix(self, view):
        self._shader['u_view'] = as_float_array(view)

    def set_projection_matrix(self, proj):
        self._shader['u_projection'] = as_float_array(proj)

    def draw(self):
        gloo.set_line_width(self._linewidth)
        self._shader.draw('line_strip')

PolylineShader.VERT_SHADER = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec3 position;
attribute vec3 color;

varying vec3 v_color;

void main(void)
{
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
    v_color = color;
}
"""

PolylineShader.FRAG_SHADER = """
#version 120

varying vec3 v_color;

void main()
{
    gl_FragColor = vec4(v_color, 1.0);
}
"""


class MeshShader(object):
    """ Shader program for rendering triangular mesh object with diffusive
    reflection of directional light.
    """
    def __init__(self, mesh):
        self._mesh = mesh
        self._shader = gloo.Program(self.VERT_SHADER, self.FRAG_SHADER)

        self._shader['position'] = as_float_array(mesh.vertices)
        self._shader['normal'] = as_float_array(mesh.normals)
        self._shader['color'] = as_float_array(mesh.colors)
        self._faces = gloo.IndexBuffer(mesh.faces)

        self.set_light(self.DEFAULT_LIGHT)
        self.set_shininess(self.DEFAULT_SHININESS)

        I = np.eye(4, dtype=np.float32)
        self.set_model_matrix(I)
        self.set_view_matrix(I)
        self.set_projection_matrix(I)

    def set_light(self, light):
        self._shader['u_light'] = as_float_array(light)

    def set_shininess(self, shininess):
        self._shader['u_shininess'] = shininess

    def set_model_matrix(self, mmat):
        self._shader['u_model'] = as_float_array(mmat)

    def set_view_matrix(self, view):
        self._shader['u_view'] = as_float_array(view)

    def set_projection_matrix(self, proj):
        self._shader['u_projection'] = as_float_array(proj)

    def draw(self):
        self._shader.draw('triangles', self._faces)

# Default light vector. Light comes from the right side of the camera.
MeshShader.DEFAULT_LIGHT = (-1 / math.sqrt(3),
                            -1 / math.sqrt(3),
                             1 / math.sqrt(3))

MeshShader.DEFAULT_SHININESS = 0.5

MeshShader.VERT_SHADER = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec3 position;
attribute vec3 normal;
attribute vec3 color;

varying vec3 v_normal;
varying vec3 v_color;

void main(void)
{
    mat4 mvp = u_projection * u_view * u_model;
    mat3 mvp_notrans = mat3(mvp);
    gl_Position = mvp * vec4(position, 1.0);
    v_normal = normalize(mvp_notrans * normal);
    v_color = color;
}
"""

MeshShader.FRAG_SHADER = """
#version 120

uniform vec3 u_light;
uniform float u_shininess;

varying vec3 v_normal;
varying vec3 v_color;

void main()
{
    /* Material color */
    vec3 material_color = v_color;
    float channel_mean = 0.33 * (material_color.r + material_color.g + material_color.b);
    vec3 material_gray = vec3(channel_mean, channel_mean, channel_mean);
    vec3 shadow_color = mix(material_color, material_gray, 0.3);

    /* Diffusion */
    float diffusion = max(0.0, -dot(u_light, v_normal));
    vec3 diffusive_color = diffusion * material_color;

    /* Dissipation */
    float dissipation_start_depth = 8.0;
    float dissipativity = 0.01;
    float dissipation_bound = 0.1;
    float depth = gl_FragCoord.z / gl_FragCoord.w;
    float mod_depth = max(depth - dissipation_start_depth, 0.0);
    float dissipation = max(exp(-dissipativity * mod_depth * mod_depth), dissipation_bound);

    /* Composite */
    vec3 base_color = mix(shadow_color, diffusive_color, u_shininess);
    vec3 color = dissipation * base_color;
    gl_FragColor = vec4(color, 1.0);
}
"""


class WireframeShader(object):
    """ Shader program for rendering triangular mesh object as wireframe.
    """
    def __init__(self, mesh, opacity=1):
        self._shader = gloo.Program(self.VERT_SHADER, self.FRAG_SHADER)
        self._shader['u_opacity'] = opacity
        self._shader['position'] = as_float_array(mesh.vertices)
        self._shader['color'] = as_float_array(mesh.colors)
        self._edges = gloo.IndexBuffer(extract_edges(mesh.faces))

        I = np.eye(4, dtype=np.float32)
        self.set_model_matrix(I)
        self.set_view_matrix(I)
        self.set_projection_matrix(I)

    def set_model_matrix(self, mmat):
        self._shader['u_model'] = as_float_array(mmat)

    def set_view_matrix(self, view):
        self._shader['u_view'] = as_float_array(view)

    def set_projection_matrix(self, proj):
        self._shader['u_projection'] = as_float_array(proj)

    def draw(self):
        self._shader.draw('lines', self._edges)


def extract_edges(faces):
    edges = []
    for i, j, k in faces:
        edges.append(tuple(sorted((i, j))))
        edges.append(tuple(sorted((j, k))))
        edges.append(tuple(sorted((k, i))))
    return sorted(list(set(edges)))


WireframeShader.VERT_SHADER = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec3 position;
attribute vec3 color;

varying vec3 v_color;

void main(void)
{
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
    v_color = color;
}
"""

WireframeShader.FRAG_SHADER = """
#version 120

uniform float u_opacity;
varying vec3 v_color;

void main()
{
    gl_FragColor = vec4(v_color, u_opacity);
}
"""


class TextureDisplayShader(object):
    """ Shader program for displaying texture on the entire viewport
    """
    def __init__(self, texture):
        # (x,y) coordinates of the quad on which the texture is rendered
        quad_vertices = as_float_array([
                (-1, -1), ( 1, -1),
                (-1,  1), ( 1,  1)
            ])
        self._shader = gloo.Program(TextureDisplayShader.VERT_SHADER,
                                    TextureDisplayShader.FRAG_SHADER)
        self._shader['u_texture'] = texture
        self._shader['position'] = quad_vertices

    def draw(self):
        self._shader.draw('triangle_strip')


TextureDisplayShader.VERT_SHADER = """
#version 120
attribute vec2 position;
varying vec2 v_texCoord;
void main(void)
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texCoord = vec2(0.5, 0.5) + 0.5 * position;
}
"""

TextureDisplayShader.FRAG_SHADER = """
#version 120
uniform sampler2D u_texture;
varying vec2 v_texCoord;
void main(void)
{
    gl_FragColor = vec4(texture2D(u_texture, v_texCoord).rgb, 1.0);
}
"""


class CompositeShader(object):
    """ Virtual shader for sequentially execute multiple shaders in one go.
    """
    def __init__(self, shaders):
        self.shaders = shaders

    def set_light(self, light):
        for shader in self.shaders:
            shader.set_light(light)

    def set_model_matrix(self, mmat):
        for shader in self.shaders:
            shader.set_model_matrix(mmat)

    def set_view_matrix(self, view):
        for shader in self.shaders:
            shader.set_view_matrix(view)

    def set_projection_matrix(self, proj):
        for shader in self.shaders:
            shader.set_projection_matrix(proj)

    def draw(self):
        for shader in self.shaders:
            shader.draw()
