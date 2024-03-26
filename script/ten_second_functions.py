import os 
import subprocess
import shutil
from PIL import Image, ImageSequence

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
    


if __name__ == '__main__':
    tripo_image_to_3d('TripoSR/examples/chair.png', 'chair')

