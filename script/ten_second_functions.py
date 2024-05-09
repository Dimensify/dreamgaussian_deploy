import os 
import subprocess
import shutil
from PIL import Image, ImageSequence
from mvdream.model_zoo import build_model
import torch
import numpy as np
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model

## Fetching the model and sampler globally to avoid reload
model = build_model("sd-v2.1-base-4view")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.device = device
model.to(device)
model.eval()
sampler = DDIMSampler(model)
uc = model.get_learned_conditioning( [""] ).to(device)


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

    ## Rendering to a gif
    os.system(f"python -m kiui.render logs/{obj_name}/{obj_name}.obj --save_video logs/{obj_name}/{obj_name}.gif --wogui --force_cuda_rast")
    ## Make the gif loop infinitely
    make_gif_loop_infinitely(f'logs/{obj_name}/{obj_name}.gif', f'logs/{obj_name}/{obj_name}.gif')

    return f'logs/{obj_name}/{obj_name}.obj', f'logs/{obj_name}/{obj_name}.gif'
    

def tripo_text_to_3d(prompt, obj_name):
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
    global model, device, sampler, uc
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
    return tripo_image_to_3d(f'logs/{obj_name}/image.png', obj_name)

if __name__ == '__main__':
    prompt = input("Enter a prompt: ")
    tripo_text_to_3d(prompt, 'test_obj')

