from collections import defaultdict
import numpy as np
from pen_world.P3.RobinsonTriangle import psi, psi2
from pen_world.P3.BtileL import BtileL
from typing import Tuple
import math

# A small tolerance for comparing floats for equality
TOL = 1.e-5

class PenroseP3:
    """ A class representing the P3 Penrose tiling. """

    def __init__(self, scale=200, ngen=4, config={}, graph=False):
        """
        Initialise the PenroseP3 instance with a scale determining the size
        of the final image and the number of generations, ngen, to inflate
        the initial triangles. Further configuration is provided through the
        key, value pairs of the optional config dictionary.

        """

        self.scale = scale
        self.ngen = ngen

        # Default configuration
        self.config = {'width': '100%', 'height': '100%',
                       'stroke-colour': '#fff',
                       'base-stroke-width': 0.05,
                       'margin': 1.05,
                       'tile-opacity': 0.6,
                       'random-tile-colours': False,
                       'Stile-colour': '#08f',
                       'Ltile-colour': '#0035f3',
                       'Aarc-colour': '#f00',
                       'Carc-colour': '#00f',
                       'draw-tiles': True,
                       'draw-arcs': False,
                       'reflect-x': True,
                       'draw-rhombuses': True,
                       'rotate': 0,
                       'flip-y': False, 'flip-x': False,
                       }
        self.config.update(config)
        # And ensure width, height values are strings for the SVG
        self.config['width'] = str(self.config['width'])
        self.config['height'] = str(self.config['height'])

        self.graph = graph
        self.elements = []
        self.vertex_to_polygons = defaultdict(list)

    def set_initial_tiles(self, tiles):
        self.elements = tiles

    def inflate(self, last=False):
        """ "Inflate" each triangle in the tiling ensemble."""
        new_elements = []
        for element in self.elements:
            new_elements.extend(element.inflate())
        self.elements = new_elements

    def remove_dupes(self):
        """
        Remove triangles giving rise to identical rhombuses from the
        ensemble.

        """

        # Triangles give rise to identical rhombuses if these rhombuses have
        # the same centre.
        selements = sorted(self.elements, key=lambda e: (e.centre().real,
                                                         e.centre().imag))
        self.elements = [selements[0]]
        for i, element in enumerate(selements[1:], start=1):
            if abs(element.centre() - selements[i-1].centre()) > TOL:
                self.elements.append(element)

    def add_conjugate_elements(self):
        """ Extend the tiling by reflection about the x-axis. """

        self.elements.extend([e.conjugate() for e in self.elements])

    def rotate(self, theta):
        """ Rotate the figure anti-clockwise by theta radians."""

        rot = np.cos(theta) + 1j * np.sin(theta)
        for e in self.elements:
            e.A *= rot
            e.B *= rot
            e.C *= rot

    def flip_y(self):
        """ Flip the figure about the y-axis. """

        for e in self.elements:
            e.A = complex(-e.A.real, e.A.imag)
            e.B = complex(-e.B.real, e.B.imag)
            e.C = complex(-e.C.real, e.C.imag)

    def flip_x(self):
        """ Flip the figure about the x-axis. """

        for e in self.elements:
            e.A = e.A.conjugate()
            e.B = e.B.conjugate()
            e.C = e.C.conjugate()

    def make_tiling(self):
        """ Make the Penrose tiling by inflating ngen times. """

        for gen in range(self.ngen):
            self.inflate()
        if self.config['draw-rhombuses']:
            self.remove_dupes()
        if self.config['reflect-x']:
            self.add_conjugate_elements()
            self.remove_dupes()

        # Rotate the figure anti-clockwise by theta radians.
        theta = self.config['rotate']
        if theta:
            self.rotate(theta)

        # Flip the image about the y-axis (note this occurs _after_ any
        # rotation.
        if self.config['flip-y']:
            self.flip_y()

        # Flip the image about the x-axis (note this occurs _after_ any
        # rotation and after any flip about the y-axis.
        if self.config['flip-x']:
            self.flip_x()

        self.collect_vertices()

        if self.graph:
            self.generate_graph()

    def get_tile_colour(self, e):
        """ Return a HTML-style colour string for the tile. """

        if self.config['random-tile-colours']:
            # Return a random colour as '#xxx'
            return '#' + hex(random.randint(0,0xfff))[2:]

        # Return the colour string, or call the colour function as appropriate
        if isinstance(e, BtileL):
            if hasattr(self.config['Ltile-colour'], '__call__'):
                return self.config['Ltile-colour'](e)
            return self.config['Ltile-colour']

        if hasattr(self.config['Stile-colour'], '__call__'):
            return self.config['Stile-colour'](e)
        return self.config['Stile-colour']

    def make_svg(self):
        """ Make and return the SVG for the tiling as a str. """

        xmin = ymin = -self.scale * self.config['margin']
        width =  height = 2*self.scale * self.config['margin']
        viewbox ='{} {} {} {}'.format(xmin, ymin, width, height)
        svg = ['<?xml version="1.0" encoding="utf-8"?>',
               '<svg width="{}" height="{}" viewBox="{}"'
               ' preserveAspectRatio="xMidYMid meet" version="1.1"'
               ' baseProfile="full" xmlns="http://www.w3.org/2000/svg">'
               .format(self.config['width'], self.config['height'], viewbox)]
        # The tiles' stroke widths scale with ngen
        stroke_width = str(psi**self.ngen * self.scale *
                           self.config['base-stroke-width'])
        svg.append('<g style="stroke:{}; stroke-width: {};'
                   ' stroke-linejoin: round;">'
                   .format(self.config['stroke-colour'], stroke_width))
        draw_rhombuses = self.config['draw-rhombuses']
        for e in self.elements:
            if self.config['draw-tiles']:
                svg.append('<path fill="{}" fill-opacity="{}" d="{}"/>'
                           .format(self.get_tile_colour(e),
                                   self.config['tile-opacity'],
                                   e.path(rhombus=draw_rhombuses)))
            if self.config['draw-arcs']:
                arc1_d, arc2_d = e.arcs(half_arc=not draw_rhombuses)
                svg.append('<path fill="none" stroke="{}" d="{}"/>'
                           .format(self.config['Aarc-colour'], arc1_d))
                svg.append('<path fill="none" stroke="{}" d="{}"/>'
                           .format(self.config['Carc-colour'], arc2_d))
        svg.append('</g>\n</svg>')
        return '\n'.join(svg)

    def write_svg(self, filename):
        """ Make and write the SVG for the tiling to filename. """
        svg = self.make_svg()
        with open(filename, 'w') as fo:
            fo.write(svg)

    def collect_vertices(self):
        # collect all robinson triangles that meet at a vertex
        for element in self.elements:
            self.vertex_to_polygons[element.A].append(element.type)
            self.vertex_to_polygons[element.B].append(element.type)
            self.vertex_to_polygons[element.C].append(element.type)

    def calculate_entropy(self):
        # count number of vertices with only reflection symmetry and number with a 5 fold symmetry (stars)
        reflection_vertices = 0
        star_vertices = 0

        # iterate through all vertices
        for polygon_list in self.vertex_to_polygons.values():

            # has B_S robinson triangle with this vertex
            has_B_S = "B_S" in polygon_list

            if has_B_S:
                reflection_vertices += 1
            else:
                star_vertices += 1

        # total number of symmetry operations
        # (reflection and identity for reflection_vertices and 5 rotations and 5 reflections for star_vertices)
        N_G = 2 * reflection_vertices + 10 * star_vertices

        # counts of different symmetry operations (ops)
        identity_ops = reflection_vertices + star_vertices
        star_rotation_ops = star_vertices
        star_reflections_ops = star_vertices
        single_reflection_ops = reflection_vertices + star_vertices

        # repeat rotations and reflection ops 4 times because we do not count the identity or single mirror plane ops
        m_G_i = [identity_ops, single_reflection_ops] + [star_rotation_ops, star_reflections_ops] * 4
        p_G_i = np.array(m_G_i) / N_G

        return -np.sum(p_G_i * np.log(p_G_i))

    def to_tuple(self, complex: complex) -> Tuple[int, int]:
        """convert complex number to tuple

        Args:
            complex (complex): complex number x + iy that represents point

        Returns:
            Tuple[int, int]: (x, y) tuple that represents point
        """
        return (complex.real, complex.imag)
        
        

    def generate_graph(self):
        """
        Generates graph by creating Vertex objects out of each point and connecting to others in each triangle
        """
        vertex_graph = defaultdict(list)
        for i, element in enumerate(self.elements):
            A, B, C, D = element.rhombus() # vertices of the rhombus that corresponds to the Robinson Triangle
            A = self.to_tuple(A)
            B = self.to_tuple(B)
            C = self.to_tuple(C)
            D = self.to_tuple(D)
            
            # add points if they aren't already added to dictionary

            # add connections to A
            if B not in vertex_graph[A]:
                # print("Adding 1: ")
                vertex_graph[A].append(B)

            if D not in vertex_graph[A]:
                vertex_graph[A].append(D)

            # add connections to B
            if A not in vertex_graph[B]:
                vertex_graph[B].append(A)

            if C not in vertex_graph[B]:
                vertex_graph[B].append(C)

            # add connections to C
            if B not in vertex_graph[C]:
                vertex_graph[C].append(B)

            if D not in vertex_graph[C]:
                vertex_graph[C].append(D)

            # add connections to D
            if A not in vertex_graph[D]:
                vertex_graph[D].append(A)

            if C not in vertex_graph[D]:
                vertex_graph[D].append(C)

        self.vertex_graph = vertex_graph

    
def generate_tiling(ngen):
    # A star with five-fold symmetry

    # The Golden ratio
    phi = 1 / psi
    scale = 100
    config = {'draw-arcs': True,
              'Aarc-colour': '#ff5e25',
              'Carc-colour': 'none',
              'Stile-colour': '#090',
              'Ltile-colour': '#9f3',
              'rotate': math.pi/2}

    tiling = PenroseP3(scale*2, ngen=ngen, config=config, graph=True)
    theta = 2*math.pi / 5
    rot = math.cos(theta) + 1j*math.sin(theta)

    B1 = scale
    p = B1 * rot
    q = p*rot

    C5 = -scale * phi
    r = C5 / rot
    s = r / rot
    A = [0]*5
    B = [scale, p, p, q, q]
    C = [s, s, r, r, C5]

    tiling.set_initial_tiles([BtileL(*v) for v in zip(A, B, C)])
    tiling.make_tiling()
    graph = tiling.vertex_graph

    array_graph = {}
    for position in graph:
        array_graph[position] = np.array(graph[position])

    return array_graph