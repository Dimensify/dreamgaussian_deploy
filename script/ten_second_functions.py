import os 
import subprocess
import shutil
from PIL import Image, ImageSequence
from mvdream.model_zoo import build_model
import torch
import numpy as np
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
import sys
import glob
import bpy 

# crm_directory = os.path.join(os.getcwd(), 'CRM')
# sys.path.append(crm_directory)
# from run import *

## Fetching the model and sampler globally to avoid reload
# print("### LOADING MVDREAM ###")
# model = build_model("sd-v2.1-base-4view")
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.device = device
# model.to(device)
# model.eval()
# sampler = DDIMSampler(model)
# uc = model.get_learned_conditioning( [""] ).to(device)

### Creating CRM Pipeline
# print("### LOADING CRM ###")
# crm_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")
# specs = json.load(open(f"{crm_directory}/configs/specs_objaverse_total.json"))
# crm_model = CRM(specs).to("cuda")
# crm_model.load_state_dict(torch.load(crm_path, map_location = "cuda"), strict=False)
# stage1_config = OmegaConf.load(f"{crm_directory}/configs/nf7_v3_SNR_rd_size_stroke.yaml").config
# stage2_config = OmegaConf.load(f"{crm_directory}/configs/stage2-v2-snr.yaml").config
# stage2_sampler_config = stage2_config.sampler
# stage1_sampler_config = stage1_config.sampler
# stage1_model_config = stage1_config.models
# stage2_model_config = stage2_config.models
# stage1_model_config.config = crm_directory + '/' + stage1_model_config.config
# stage2_model_config.config = crm_directory + '/' + stage2_model_config.config
# xyz_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="ccm-diffusion.pth")
# pixel_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth")
# stage1_model_config.resume = pixel_path
# stage2_model_config.resume = xyz_path

# pipeline = TwoStagePipeline(
#     stage1_model_config,
#     stage2_model_config,
#     stage1_sampler_config,
#     stage2_sampler_config,
# )

def convert_to_standard_obj(input_path, output_path):
    """
    Convert color on vertex .obj into standard .obj with unwrapped UV and .png texture file.
    The output files will have the same base name as the input file.

    :param input_path: Path to the input .obj file
    :param output_path: Path to the output directory
    """
    
    def clear_scene():
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
    
    clear_scene()

    # Extract the base name of the input file without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Import the .obj file
    bpy.ops.wm.obj_import(filepath=input_path, directory=os.path.split(input_path)[0], files=[{"name": os.path.split(input_path)[1]}])
    bpy.context.object.rotation_euler[0] = 0
    obj = bpy.context.active_object

    # Add UV map
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.editmode_toggle()

    # Create vertex color material
    mat = bpy.data.materials.new(name="VertexColor")
    mat.use_nodes = True
    vc = mat.node_tree.nodes.new('ShaderNodeVertexColor')
    vc.name = "vc_node"
    vc.layer_name = "Color"
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    mat.node_tree.links.new(vc.outputs[0], bsdf.inputs[0])
    obj.data.materials.append(mat)

    # Bake the texture
    image_name = base_name + '_BakedTexture'
    img = bpy.data.images.new(image_name, 1024, 1024)

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.bake_type = 'DIFFUSE'
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_selected_to_active = False

    for mat in obj.data.materials:
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        texture_node = nodes.new('ShaderNodeTexImage')
        texture_node.name = 'Bake_node'
        texture_node.select = True
        nodes.active = texture_node
        texture_node.image = img
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type='DIFFUSE', save_mode='EXTERNAL')

    # Remove the original vertex color node
    for mat in obj.data.materials:
        vc = mat.node_tree.nodes['vc_node']
        mat.node_tree.nodes.remove(vc)

    os.makedirs(output_path, exist_ok=True)

    # Save the baked texture
    img.save_render(filepath=f'{output_path}/{base_name}_texture_kd.png')

    # Export the .obj and .mtl files
    obj_output_path = f"{output_path}/{base_name}.obj"
    if bpy.app.version[0] >= 4:
        bpy.ops.wm.obj_export(filepath=obj_output_path)
    else:
        bpy.ops.export_scene.obj(filepath=obj_output_path)

    mtl_output_path = f"{output_path}/{base_name}.mtl"
    with open(mtl_output_path, 'a') as f:
        f.write('map_Kd {}_texture_kd.png'.format(base_name))

def convert_to_vertex_color_obj(input_path, output_path):
    """
    Convert standard .obj with unwrapped UV and .png texture file into color on vertex .obj.
    The output files will have the same base name as the input file.

    :param input_path: Path to the input .obj file
    :param output_path: Path to the output directory
    """
    
    def clear_scene():
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
    
    clear_scene()

    # Detect file format
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    
    # Import file
    if ext == '.obj':
        bpy.ops.import_scene.obj(filepath=input_path)
    elif ext == '.glb':
        bpy.ops.import_scene.gltf(filepath=input_path)
    else:
        raise ValueError('Unsupported file format')

    # Extract the base name of the input file without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Import the .obj file
    # bpy.ops.import_scene.obj(filepath=input_path)
    obj = bpy.context.selected_objects[0]
    
    # Load the texture image
    img_path = os.path.join(os.path.dirname(input_path), base_name + '_texture_kd.png')
    img = bpy.data.images.load(img_path)
    
    # Ensure the object is active and in object mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add a vertex color layer
    if not obj.data.vertex_colors:
        obj.data.vertex_colors.new(name='Col')
    
    vertex_colors = obj.data.vertex_colors['Col']
    
    # Create a material and assign it to the object
    mat = bpy.data.materials.new(name="TextureToVertexColor")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    
    if bsdf:
        tex_image_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image_node.image = img
        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])
    
    obj.data.materials.append(mat)
    
    # Switch to edit mode to access the mesh data
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.project_from_view()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Loop through each polygon and assign vertex colors based on the texture
    for poly in obj.data.polygons:
        for loop_index in poly.loop_indices:
            loop_vert_index = obj.data.loops[loop_index].vertex_index
            uv_coords = obj.data.uv_layers.active.data[loop_index].uv
            color = img.sample(uv_coords.x, uv_coords.y)
            vertex_colors.data[loop_index].color = color[:3]  # Assign RGB values
    
    # Remove the UV map and material
    obj.data.uv_layers.clear()
    obj.data.materials.clear()
    
    os.makedirs(output_path, exist_ok=True)

    # Export the .obj file
    obj_output_path = os.path.join(output_path, base_name + '_vertex_color.obj')
    bpy.ops.export_scene.obj(filepath=obj_output_path)

def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1):
    '''
    Converts text to image

    Parameters
    ----------
    model: torch.nn.Module
        The model to be used for text to image conversion
    image_size: int
        Size of the image
    '''
    if type(prompt)!=list:
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size,1,1)}
        uc_ = {"context": uc.repeat(batch_size,1,1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=None)
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


def make_gif_loop_infinitely(input_gif_path, output_gif_path):
    '''
    Modifies the loop flag of a GIF file to make it loop infinitely

    Parameters
    ----------
    input_gif_path: str
        Path to the input GIF file
    output_gif_path: str    
        Path to the output GIF file

    Returns
    -------
    None
    '''
    # Open the GIF file
    gif = Image.open(input_gif_path)

    frames = []
    for frame in ImageSequence.Iterator(gif):
        frames.append(frame.copy())

    # Modify the loop flag to make the GIF loop infinitely
    if len(frames) > 1:
        # Setting the loop flag to 0 will make the GIF loop indefinitely
        frames[0].info['loop'] = 0

    # Save the modified frames as a new GIF file
    frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], loop=0, duration=gif.info['duration'])

def crm_image_to_3d(path, obj_name):
    '''
    Converts an image to a 3D object and renders it as a GIF

    Parameters
    ----------
    path: str
        Path to the image file
    obj_name: str
        Name of the object

    Returns
    -------
    str
        Path to the 3D object file
    str
        Path to the GIF file
    '''
    path = os.path.abspath(path)
    logpath = os.path.abspath(f'./logs')
    ## Run the command to convert the image to 3D: python run.py examples/chair.png --output-dir output/ --model-save-format glb
    command = f'python run.py --inputdir {path} --outdir {logpath}/{obj_name}/'
    subprocess.run(command, shell=True, cwd="./CRM")
    ## unzip output3d.zip
    os.system(f"unzip {logpath}/{obj_name}/output3d.zip -d {logpath}/{obj_name}/")
    temp_file_name = glob.glob(f'logs/{obj_name}/*.obj')[0].split('/')[-1].split('.')[0]

    ## Removing the logs
    os.remove(f'{logpath}/{obj_name}/pixel_images.png')
    os.remove(f'{logpath}/{obj_name}/preprocessed_image.png')
    os.remove(f'{logpath}/{obj_name}/xyz_images.png')
    os.remove(f'{logpath}/{obj_name}/output3d.zip')

    ## Rendering to a gif
    os.system(f"python -m kiui.render {logpath}/{obj_name}/{temp_file_name}.obj --save_video {logpath}/{obj_name}/{obj_name}.gif --wogui --force_cuda_rast")
    ## Make the gif loop infinitely
    make_gif_loop_infinitely(f'{logpath}/{obj_name}/{obj_name}.gif', f'{logpath}/{obj_name}/{obj_name}.gif')

    return f'logs/{obj_name}/{temp_file_name}.obj', f'logs/{obj_name}/{obj_name}.gif'
    

def tripo_image_to_3d(path, obj_name):
    '''
    Converts an image to a 3D object and renders it as a GIF

    Parameters
    ----------
    path: str
        Path to the image file
    obj_name: str
        Name of the object

    Returns
    -------
    str
        Path to the 3D object file
    str
        Path to the GIF file
    '''
    ## Run the command to convert the image to 3D: python run.py examples/chair.png --output-dir output/ --model-save-format glb
    command = f'python TripoSR/run.py {path} --output-dir logs/{obj_name}/ --model-save-format obj'
    subprocess.run(command, shell=True, cwd="./")
    ## Moveing mesh.obj and render.mp4 from logs/{obj_name}/0 to logs/{obj_name}/ and renaming them
    os.rename(f'logs/{obj_name}/0/mesh.obj', f'logs/{obj_name}/{obj_name}.obj') 
    ## Removing the logs/{obj_name}/0 directory
    shutil.rmtree(f'logs/{obj_name}/0')

    ## Converting the color on vertex .obj to standard .obj
    convert_to_standard_obj(f'logs/{obj_name}/{obj_name}.obj', f'logs/{obj_name}')

    ## Rendering to a gif
    os.system(f"python -m kiui.render logs/{obj_name}/{obj_name}.obj --save_video logs/{obj_name}/{obj_name}.gif --wogui --force_cuda_rast")
    ## Make the gif loop infinitely
    make_gif_loop_infinitely(f'logs/{obj_name}/{obj_name}.gif', f'logs/{obj_name}/{obj_name}.gif')

    return f'logs/{obj_name}/{obj_name}.obj', f'logs/{obj_name}/{obj_name}.gif'
    

def text_to_3d(prompt, obj_name, method='tripo'):
    '''
    Converts a text to a 3D object and renders it as a GIF

    Parameters
    ----------
    prompt: str
        Text prompt
    obj_name: str
        Name of the object

    Returns
    -------
    str
        Path to the 3D object file
    str
        Path to the GIF file
    '''
    # global model, device, sampler, uc
    print("### LOADING MVDREAM ###")
    model = build_model("sd-v2.1-base-4view")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.device = device
    model.to(device)
    model.eval()
    sampler = DDIMSampler(model)
    uc = model.get_learned_conditioning( [""] ).to(device)
    dtype = torch.float16
    camera = None
    batch_size = 1
    prompt = prompt + '.3D model, White background, symmetric, front facing.'

    img = t2i(model, 256, prompt, uc, sampler, step=100, scale=10, batch_size=batch_size, ddim_eta=0.0, 
            dtype=dtype, device=device, camera=camera, num_frames=4)
    img = np.concatenate(img, 1)
    ## Save the image to logs/{obj_name}/image.png
    img = Image.fromarray(img)
    ## make the logs/{obj_name} directory
    os.makedirs(f'logs/{obj_name}', exist_ok=True)
    img.save(f'logs/{obj_name}/image.png')

    ## Use image to 3D function
    if method == 'tripo':
        return tripo_image_to_3d(f'logs/{obj_name}/image.png', obj_name)
    elif method == 'crm':
        return crm_image_to_3d(f'logs/{obj_name}/image.png', obj_name)

if __name__ == '__main__':
    prompt = input("Enter a prompt: ")
    text_to_3d(prompt, 'test_obj', method='tripo')
    # crm_image_to_3d('CRM/examples/kunkun.webp','testobj')

