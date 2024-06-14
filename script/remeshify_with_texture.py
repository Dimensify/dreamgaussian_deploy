import pymeshlab as pml


input_mesh_path = "/home/ashu/checkouts/threeflow/assets/crossflow_turbine_with_mtl/model.glb"
output_mesh_path = "/home/ashu/checkouts/threeflow/assets/crossflow_turbine_with_mtl/model_out.obj"


ms = pml.MeshSet()
ms.load_new_mesh(input_mesh_path)
ms.apply_filter('meshing_tri_to_quad_by_4_8_subdivision')
ms.load_new_mesh(input_mesh_path)
ms.apply_filter('transfer_texture_to_color_per_vertex', sourcemesh=0, targetmesh=1)

ms.save_current_mesh(output_mesh_path, save_textures=True)