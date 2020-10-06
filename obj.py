# Carga un archivo OBJ
import struct
import numpy as np
from gl_aux import *

def color(r, g, b):
    return bytes([int(b * 255), int(g * 255), int(r * 255)])

class Obj(object):
    def __init__(self, filename):
        #open(filename, encoding="utf8")
        with open(filename, 'r', encoding="utf8") as file:
            self.lines = file.read().splitlines()

        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        self.read()

    def read(self):
        for line in self.lines:
            if line:
                if len(line) > 0 :
                    if line[0] != '#' :
                        
                        prefix, value = line.split(' ', 1)
                        value = value.strip().replace('//','/').replace('  ',' ')

                        if prefix == 'v':
                            self.vertices.append(list(map(float,value.split(' '))))
                        elif prefix == 'f':
                            nvalue = value.strip()
                            self.faces.append([list(map(int, face.split('/'))) for face in nvalue.split(' ')])
                        elif prefix == 'vn':
                            nvalue = value.strip()
                            self.normals.append(list(map(float,value.split(' '))))
                        elif prefix == 'vt':
                            nvalue = value.strip()
                            self.texcoords.append(list(map(float,value.split(' '))))

class Texture(object):
    def __init__(self, path):
        self.path = path
        self.read()
        
    def read(self):
        image = open(self.path, 'rb')
        image.seek(10)
        headerSize = struct.unpack('=l', image.read(4))[0]

        image.seek(14 + 4)
        self.width = struct.unpack('=l', image.read(4))[0]
        self.height = struct.unpack('=l', image.read(4))[0]
        image.seek(headerSize)

        self.pixels = []

        for y in range(self.height):
            self.pixels.append([])
            for x in range(self.width):
                b = ord(image.read(1)) / 255
                g = ord(image.read(1)) / 255
                r = ord(image.read(1)) / 255
                self.pixels[y].append(color(r,g,b))

        image.close()

    def getColor(self, tx, ty):
        if tx >= 0 and tx <= 1 and ty >= 0 and ty <= 1:
            x = int(tx * self.width - 1)
            y = int(ty * self.height - 1)

            return self.pixels[y][x]
        else:
            return color(0,0,0)

class Envmap(object):
    def __init__(self, path):
        self.path = path
        self.read()
        
    def read(self):
        image = open(self.path, 'rb')
        image.seek(10)
        headerSize = struct.unpack('=l', image.read(4))[0]

        image.seek(14 + 4)
        self.width = struct.unpack('=l', image.read(4))[0]
        self.height = struct.unpack('=l', image.read(4))[0]
        image.seek(headerSize)

        self.pixels = []

        for y in range(self.height):
            self.pixels.append([])
            for x in range(self.width):
                b = ord(image.read(1)) / 255
                g = ord(image.read(1)) / 255
                r = ord(image.read(1)) / 255
                self.pixels[y].append(color(r,g,b))

        image.close()

    def getColor(self, direction):

        direction_normal = vectNormal(direction)
        direction = V3(direction.x / direction_normal, direction.y / direction_normal, direction.z / direction_normal)

        x = int( (np.arctan2( direction.z, direction.x) / (2 * np.pi) + 0.5) * self.width)
        y = int( np.arccos(-direction.y) / np.pi * self.height )

        return self.pixels[y][x]