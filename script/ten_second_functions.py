import os 
import subprocess
import shutil

def tripo_image_to_3d(path, obj_name):
    '''
    
    '''
    ## Run the command to convert the image to 3D: python run.py examples/chair.png --output-dir output/ --model-save-format glb
    command = f'python TripoSR/run.py {path} --output-dir logs/{obj_name}/ --model-save-format obj --render'
    subprocess.run(command, shell=True, cwd="./")
    ## Moveing mesh.obj and render.mp4 from logs/{obj_name}/0 to logs/{obj_name}/ and renaming them
    os.rename(f'logs/{obj_name}/0/mesh.obj', f'logs/{obj_name}/{obj_name}.obj') 
    os.rename(f'logs/{obj_name}/0/render.mp4', f'logs/{obj_name}/{obj_name}.mp4')
    ## Removing the logs/{obj_name}/0 directory
    shutil.rmtree(f'logs/{obj_name}/0')

    ## Convert MP4 to GIF
    command = f'ffmpeg -i logs/{obj_name}/{obj_name}.mp4 -vf "fps=10,scale=320:-1:flags=lanczos" logs/{obj_name}/{obj_name}.gif'
    subprocess.run(command, shell=True, cwd="./")

    ## Remove the MP4 file
    os.remove(f'logs/{obj_name}/{obj_name}.mp4')

    return f'logs/{obj_name}/{obj_name}.obj', f'logs/{obj_name}/{obj_name}.gif'
    


if __name__ == '__main__':
    tripo_image_to_3d('TripoSR/examples/chair.png', 'chair')

