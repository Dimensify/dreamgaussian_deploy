import numpy as np
import pymeshlab as pml

# https://github.com/jiawei-ren/dreamgaussian4d/blob/main/mesh_utils.py
# https://stackoverflow.com/questions/65419221/how-to-use-pymeshlab-to-reduce-vertex-number-to-a-certain-number
# https://gist.github.com/tylerlindell/7435ca2261e7c404ccc1241f18e483aa
def poisson_mesh_reconstruction(points, normals=None):
    # points/normals: [N, 3] np.ndarray

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # outlier removal
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10)

    # normals
    if normals is None:
        pcd.estimate_normals()
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals[ind])

    # visualize
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # visualize
    o3d.visualization.draw_geometries([mesh])

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(
        f"[INFO] poisson mesh reconstruction: {points.shape} --> {vertices.shape} / {triangles.shape}"
    )

    return vertices, triangles


def decimate_mesh(
    verts, faces, target, backend="pymeshlab", remesh=False, optimalplacement=True
):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.Percentage(1))
        # ms.meshing_decimation_quadric_edge_collapse(
        #     targetfacenum=int(target), optimalplacement=optimalplacement
        # )
        ms.simplification_quadric_edge_collapse_decimation(
            targetfacenum=int(target), optimalplacement=optimalplacement
        )

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            # ms.meshing_isotropic_explicit_remeshing(
            #     iterations=3, targetlen=pml.Percentage(1)
            # )
            ms.remeshing_isotropic_explicit_remeshing(
                iterations=3, 
                targetlen=pml.Percentage(1)
            )

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(
        f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces


def clean_mesh(
    verts,
    faces,
    v_pct=1,
    min_f=64,
    min_d=20,
    repair=True,
    remesh=True,
    remesh_size=0.01,
):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    # ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces
    ms.remove_unreferenced_vertices()

    if v_pct > 0:
        # ms.meshing_merge_close_vertices(
        #     threshold=pml.Percentage(v_pct)
        # )  # 1/10000 of bounding box diagonal
        ms.merge_close_vertices(
            threshold=pml.Percentage(v_pct)
            )

    # ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.remove_duplicate_faces()
    # ms.meshing_remove_null_faces()  # faces with area == 0
    ms.remove_zero_area_faces()

    if min_d > 0:
        # ms.meshing_remove_connected_component_by_diameter(
        #     mincomponentdiag=pml.Percentage(min_d)
        # )
        ms.remove_isolated_pieces_wrt_diameter(
            mincomponentdiag=pml.Percentage(min_d)
        )

    if min_f > 0:
        # ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)
        ms.remove_isolated_pieces_wrt_face_num(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        # ms.meshing_repair_non_manifold_edges(method=0)
        ms.repair_non_manifold_edges_by_removing_faces()
        # ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        ms.repair_non_manifold_vertices_by_splitting(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        # ms.meshing_isotropic_explicit_remeshing(
        #     iterations=3, targetlen=pml.AbsoluteValue(remesh_size)
        # )
         ms.remeshing_isotropic_explicit_remeshing(
            iterations=3, 
            targetlen=pml.Percentage(1)
        )


    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(
        f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces


def reduce_vertices(input_mesh_ply, output_mesh_ply):
    ms = pml.MeshSet()
    ms.load_new_mesh(input_mesh_ply)
    m = ms.current_mesh()
    print('input mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')

    #Target number of vertex
    TARGET=10000

    #Estimate number of faces to have 100+10000 vertex using Euler
    numFaces = 100 + 2*TARGET

    #Simplify the mesh. Only first simplification will be agressive
    while (ms.current_mesh().vertex_number() > TARGET):
        ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=True)
        print("Decimated to", numFaces, "faces mesh has", ms.current_mesh().vertex_number(), "vertex")
        #Refine our estimation to slowly converge to TARGET vertex number
        numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)

    m = ms.current_mesh()
    print('output mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')
    ms.save_current_mesh(output_mesh_ply)
    

def example_apply_filter(input_mesh_path, output_mesh_path):
    # create a new MeshSet
    ms = pml.MeshSet()

    # load mesh
    ms.load_new_mesh(input_mesh_path)

    # apply convex hull filter to the current selected mesh (last loaded)
    ms.generate_convex_hull()
    # alternatively:
    # ms.apply_filter('generate_convex_hull')

    assert ms.mesh_number() == 2

    # save the current selected mesh
    ms.save_current_mesh(output_mesh_path)

    # get a reference to the current selected mesh
    m = ms.current_mesh()

    print(m.vertex_number())

    assert m.vertex_number() == 455


def trimesh_to_quadmesh(input_mesh_ply, output_mesh_ply):
    ms = pml.MeshSet()
    ms.load_new_mesh(input_mesh_ply)
    
    ms.apply_filter('meshing_tri_to_quad_by_4_8_subdivision')
    # ms.apply_filter('meshing_tri_to_quad_by_smart_triangle_pairing')
    # Possible enum values for level:
    # 0: 'Fewest triangles'
    # 1: '(in between)'
    # 2: 'Better quad shape'
    # ms.apply_filter('meshing_tri_to_quad_dominant', level=0)
    
    ms.save_current_mesh(output_mesh_ply, save_textures=False)
    

