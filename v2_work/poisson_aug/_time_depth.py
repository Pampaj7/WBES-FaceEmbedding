import time, glob, numpy as np, open3d as o3d
f=sorted(glob.glob('datasets/REMESH/npz_data_topo_500/*_original.npz'))[0]
d=np.load(f); V=np.asarray(d["V"],dtype=np.float64); F=np.asarray(d["F"],dtype=np.int32)
m=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),o3d.utility.Vector3iVector(F)); m.compute_vertex_normals()
p=m.sample_points_uniformly(number_of_points=20000)
p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08,max_nn=30))
p.orient_normals_consistent_tangent_plane(20)
for depth in (7,8):
    t=time.time(); r,_=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(p,depth=depth)
    print(f"depth={depth}  {time.time()-t:7.1f}s  {len(r.triangles)} tris", flush=True)
