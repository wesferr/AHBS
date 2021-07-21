import bpy
import bmesh
import numpy as np
from mathutils import Vector
from bpy import context

plane = bpy.data.objects['Plane']
pvertices = plane.data.vertices

v1 = pvertices[0].co - pvertices[1].co
v2 = pvertices[0].co - pvertices[2].co

nplano = v1.cross(v2)
dplano = nplano.dot(-pvertices[0].co)

hashmap = {}

def calculate_plane(vertice):
    ax = nplano[0] * vertice[0]
    by = nplano[1] * vertice[1]
    cz = nplano[2] * vertice[2]
    return ax+by+cz+dplano

def test_values(a,b,c):
    if a > 0 and b > 0 and c > 0:
        return False
    elif a < 0 and b < 0 and c < 0:
        return False
    else:
        return True
    
def calculate_point(p0, p1):
    
    vd = p1 - p0
    integer_part = nplano.dot(p0)
    incognita_part = nplano.dot(vd)
    t = (dplano + integer_part)/incognita_part
    intersection_point = p0 + (-t*vd)
    return intersection_point

def baricentric_coordinates(a,b,c,p):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 =  v0.dot(v0)
    d01 =  v0.dot(v1)
    d11 =  v1.dot(v1)
    d20 =  v2.dot(v0)
    d21 =  v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0 - v - w;
    return np.array([u, v, w])
    

obj = bpy.data.objects['Body']

nmesh = bpy.data.meshes.new('Curve')
nobj = bpy.data.objects.new("Curve", nmesh)
bpy.context.collection.objects.link(nobj)

polygons = obj.data.polygons
vertices = obj.data.vertices

bm = bmesh.new()

vertice_coordinates = []

def add_to_coordinates(v1, v2, v3, i1, i2, i3):
    point = calculate_point(v2,v1)
    array = np.array([(i1,i2,i3), baricentric_coordinates(v1, v2, v3, point)])
    
    hashfunc = (i1*i2)+i1+i2
    if hashfunc not in hashmap:
        hashmap[hashfunc] = True
        bm.verts.new(point)
        vertice_coordinates.append(array.reshape(array.size))

for polygon in polygons:
    i1 = polygon.vertices[0]
    v1 = vertices[i1].co
    r1 = calculate_plane(v1)
    
    i2 = polygon.vertices[1]
    v2 = vertices[i2].co
    r2 = calculate_plane(v2)
    
    i3 = polygon.vertices[2]
    v3 = vertices[i3].co
    r3 = calculate_plane(v3)
    
    result = test_values(r1,r2,r3)
    
    if result:
        if r2 * r3 > 0:
            add_to_coordinates(v1, v2, v3, i1, i2, i3)
            add_to_coordinates(v1, v3, v2, i1, i3, i2)
            
        elif r1 * r3 > 0:
            add_to_coordinates(v2, v1, v3, i2, i1, i3)
            add_to_coordinates(v2, v3, v1, i2, i3, i1)

        elif r1 * r2 > 0:
            add_to_coordinates(v3, v2, v1, i3, i2, i1)
            add_to_coordinates(v3, v1, v2, i3, i1, i2)

            
np.save("resultado_panturrilha.npy", np.array(vertice_coordinates))
        
bm.to_mesh(nmesh)  
bm.free()