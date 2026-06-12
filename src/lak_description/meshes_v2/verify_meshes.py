import trimesh

mesh = trimesh.load("visual/head.stl")

print("centroid:", mesh.centroid)
print("bounds:", mesh.bounds)
print("extents:", mesh.extents)