import numpy as np
from numpy import matrix, cos, sin, tan

from collections import namedtuple

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
V4 = namedtuple('Point4', ['x', 'y', 'z','w'])

def vectAdd(V1 = V3(0,0,0), V2 = V3(0,0,0)):
    return V3(V1.x + V2.x, V1.y + V2.y, V1.z + V2.z)

def vectSubtract(V1 = V3(0,0,0), V2 = V3(0,0,0)):
    return V3(V1.x - V2.x, V1.y - V2.y, V1.z - V2.z)

def vectDot(V1 = V3(0,0,0), V2 = V3(0,0,0)):
    return (V1.x * V2.x) + (V1.y * V2.y) + ( V1.z * V2.z)

def vectCross(s1 = V3(0,0,0), s2 = V3(0,0,0)):
    return V3(((s1.y * s2.z)- (s2.y * s1.z)), -((s1.x * s2.z)- (s2.x * s1.z)), ((s1.x * s2.y)- (s2.x * s1.y))) 

def vectNormal(vector = V3(0,0,0)):
    result = ( abs(vector.x)**2 + abs(vector.y)**2 + abs(vector.z)**2 )**(1/2)
    return result

def invMatrix(matrix):
    return matrix**-1

def modelMatrix(translate = V3(0,0,0), scale = V3(1,1,1), rotate=V3(0,0,0)):

        ts = matrix([[1, 0, 0, translate.x],
                     [0, 1, 0, translate.y],
                     [0, 0, 1, translate.z],
                     [0, 0, 0, 1]])

        sm = matrix([[scale.x,       0,       0, 0],
                     [      0, scale.y,       0, 0],
                     [      0,       0, scale.z, 0],
                     [      0,       0,       0, 1]])

        rm =  rotationMatrix(rotate)

        return ts * rm * sm

def rotationMatrix(rotate=V3(0,0,0)):

    pitch = np.deg2rad(rotate.x)
    yaw = np.deg2rad(rotate.y)
    roll = np.deg2rad(rotate.z)

    rX = matrix([[1,          0,          0, 0],
                 [0, cos(pitch),-sin(pitch), 0],
                 [0, sin(pitch), cos(pitch), 0],
                 [0,           0,         0, 1]])

    rY = matrix([[ cos(yaw), 0, sin(yaw), 0],
                 [        0, 1,        0, 0],
                 [-sin(yaw), 0, cos(yaw), 0],
                 [        0, 0,        0, 1]])

    rZ = matrix([[cos(roll),-sin(roll), 0, 0],
                 [sin(roll), cos(roll), 0, 0],
                 [        0,         0, 1, 0],
                 [        0,         0, 0, 1]])

    return rX * rY * rZ

def viewMatrix(camPosition = V3(0,0,0), camRotation = V3(0,0,0)):
        camMatrix = modelMatrix( translate = camPosition, rotate = camRotation)
        return invMatrix(camMatrix)

def look(eye, camPosition = V3(0,0,0)):

    forward = vectSubtract(camPosition, eye)
    normal_forward = vectNormal(V3(forward[0],forward[1],forward[2]))
    forward = V3(forward[0] / normal_forward,forward[1] / normal_forward,forward[2] / normal_forward)

    right = vectCross(V3(0,1,0), V3(forward[0],forward[1],forward[2]))
    normal_right = vectNormal(V3(right[0],right[1],right[2]))
    right = V3(right[0] / normal_right,right[1] / normal_right,right[2] / normal_right)

    up = vectCross(V3(forward[0],forward[1],forward[2]), V3(right[0],right[1],right[2]))
    normal_up = vectNormal(up)
    up = V3(up.x / normal_up,up.y / normal_up,up.z / normal_up)

    camMatrix = matrix([[right[0], up[0], forward[0], camPosition.x],
                        [right[1], up[1], forward[1], camPosition.y],
                        [right[2], up[2], forward[2], camPosition.z],
                        [0,0,0,1]])

    return invMatrix(camMatrix)

def projectionMatrix(n = 0.1, f = 1000, fov = 60, width= 800, height=800):

    t = tan((fov * np.pi / 180) / 2) * n
    r = t * width / height

    return matrix([[n / r,     0,            0,              0],
                   [    0, n / t,            0,              0],
                   [    0,     0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                   [    0,     0,           -1,              0]])