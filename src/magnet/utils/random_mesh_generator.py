#import bpy
import trimesh
import pickle
import numpy as np
import stripy
import scipy
import pyglet

from scipy.interpolate import Rbf

def random_points_on_sphere(num_points: int = 10):
        points = np.random.randn(num_points, 3)
        points /= np.sum(np.abs(points)**2,axis=1, keepdims=True)**(1./2)
        return points

def fibonacci_points_on_sphere(num_points:int = 10):

    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def vertex_transform(vertices: np.ndarray, offset_values: np.ndarray):
    vertices = (1+offset_values.reshape(-1,1))*vertices

    return vertices

def triangulate_faces_from_vertices(vertices: np.ndarray):
    lons, lats = stripy.spherical.xyz2lonlat(vertices[:,0], vertices[:,1], vertices[:,2])
    faces = stripy.spherical.sTriangulation(lons, lats, permute=True).simplices
    return faces

def calculate_sphere_offset_from_knots(knot_points: np.ndarray, knot_values: np.ndarray, target_points: np.ndarray):
     rbfi = Rbf(knot_points[:,0], knot_points[:,1], knot_points[:,2], knot_values, function='gaussian')
     target_values = rbfi(target_points[:,0], target_points[:,1], target_points[:,2])
     return target_values


def generate_random_shape(base_radius: float = 1.0, sphere_subdivisions: int = 5, num_knot_points: int = 10, relative_disruption_strength: float = 0.1):
     
    sphere_vertices = trimesh.primitives.Sphere(1, subdivisions=sphere_subdivisions).vertices.copy()
    knot_vertices = fibonacci_points_on_sphere(num_knot_points)
    offsets_at_knots = relative_disruption_strength*np.random.randn(num_knot_points)

    offsets = calculate_sphere_offset_from_knots(knot_vertices, offsets_at_knots, sphere_vertices)

    new_vertices = vertex_transform(sphere_vertices, offsets)
    new_vertices = vertex_transform(new_vertices, np.array([base_radius]))
    faces = triangulate_faces_from_vertices(new_vertices)


    mesh_test = trimesh.base.Trimesh(new_vertices, faces)
    return mesh_test

mesh = generate_random_shape(num_knot_points=100, base_radius=80)
dipole_files = [f"Dipole_{i}.stl" for i in range(1, 9)]
PATH = "data/dipoles/simple"
import os
dipoles = [trimesh.load_mesh(os.path.join(PATH, 'raw', f)) for f in dipole_files]
mesh = trimesh.util.concatenate([mesh]+dipoles)
mesh.show(smooth=False)
#mesh.export('mesh.stl', file_type='stl_ascii')

quit()

someobject_bytes = pickle.dumps((32, 1))

test = xxhash.xxh64(someobject_bytes, seed=20141025).intdigest()
print(str((32, 1)))

x = range(64)
y = range(64)
x, y = np.meshgrid(x, y)



print(test)
quit()
[values[:, -pad:, :], values, values[:, :pad, :]]

# remove default scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)

t = bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))


object = bpy.data.objects['Sphere']

# bpy.context.view_layer.objects.active = bpy.data.objects['Sphere']
bpy.ops.object.modifier_add(type='WAVE')
# deform = bpy.context.object.modifiers["SimpleDeform"]

bpy.context.object.modifiers["Wave"].use_normal = True
bpy.context.object.modifiers["Wave"].width = 0.3
bpy.context.object.modifiers["Wave"].time_offset = -2

# bpy.context.object.modifiers["SimpleDeform"].deform_axis = 'Y'

# bpy.context.object.modifiers["SimpleDeform"].angle = 0.0
# bpy.context.object.modifiers["SimpleDeform"].factor = 0.7
# bpy.context.object.modifiers["SimpleDeform"].is_active = False
# bpy.ops.object.modifier_apply(modifier="SimpleDeform")

bpy.ops.export_mesh.stl(filepath="test.stl", ascii=True)



##merged_mesh = trimesh.util.concatenate(*meshe)

mesh = trimesh.load_mesh("test.stl")
mesh.show()