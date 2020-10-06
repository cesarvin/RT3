from gl import Raytracer, color, V2, V3
from obj import Obj, Texture, Envmap
from sphere import *
import random

wall = Material(diffuse = color( 0.49, 0.67, 0.48 ), spec = 16)
roof = Material(diffuse= color( 0.66, 0.84, 0.67 ), spec = 16)
floor = Material(diffuse = color(0.4, 0.35, 0.35 ), spec = 16)
cubo = Material(diffuse = color(0.4, 0.69, 0.8 ), spec = 32)


width = 500
height = 500
r = Raytracer(width,height)
r.glClearColor(0.2, 0.6, 0.8)
r.glClear()

r.pointLight = PointLight(position = V3(1,1,3), intensity = 0.75)
r.ambientLight = AmbientLight(strength = 0.1)

print('\nThis render gonna be legen—\n')

# cuarto
r.scene.append( Plane( V3(0,-15,0), V3(0,1,0), floor))
r.scene.append( Plane( V3(0,15,0), V3(0,-1,0), roof))
r.scene.append( Plane( V3(-15,0,0), V3(1,0,0), wall))
r.scene.append( Plane( V3(15,0,0), V3(-1,0,0), wall))
r.scene.append( Plane( V3(0,0,-45), V3(0,0,1), wall))

# cubos
r.scene.append( AABB(V3(0, -2.1, -10), 1.5, cubo ) )
r.scene.append( AABB(V3(1.3, 1.8, -7), 0.75, cubo ) )

r.rtRender()

print('\n—dary!\n')

r.glFinish('room.bmp')





