from fastapi import FastAPI, UploadFile, Form, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import subprocess
from mangum import Mangum
from PIL import Image, ImageSequence
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from datetime import datetime
import uvicorn
import sys 
from moviepy.editor import VideoFileClip
from glob import glob
from pathlib import Path
import yaml
import hashlib

app = FastAPI()
origins = ['https://dimensify.ai','null']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
handler = Mangum(app)
config = uvicorn.Config(app="main:app")

# Directory to store uploaded files
UPLOAD_DIR = "./ImageDream/data"

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

## Creating the output directory
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

##Config YAML File
text_to_3D_mvdream_yaml = "../configs/mvdream-sd21.yaml"
text_to_3D_shading_mvdream_yaml = "../configs/mvdream-sd21-shading.yaml"
image_text_to_3D_imagedream_yaml = "../configs/imagedream-sd21-shading.yaml"

# Add the path to PYTHONPATH
new_path = './ImageDream/extern/ImageDream'
os.environ['PYTHONPATH'] = os.pathsep.join([os.environ.get('PYTHONPATH', ''), new_path])


### UTILITIES ###
def make_gif(input_path,output_path):
    '''
    Converts the video to GIF

    Parameters
    ----------
    input_path: str
        Path to the input video
    output_path: str
        Path to the output GIF

    Returns
    -------
    str:
        Path to the output GIF
    '''

    input_video_path = input_path
    output_gif_path = output_path

    # Load the video
    video_clip = VideoFileClip(input_video_path)

    # Set the frame rate for the GIF (adjust as needed)
    frame_rate = 10

    # Resize the video to a smaller size (adjust as needed)
    target_width = 1536
    target_height = 512
    method = Image.FASTOCTREE
    colors = 250
    video_clip = video_clip.resize((target_width, target_height))
    # Create a list to store GIF frames
    gif_frames = []

    # Generate GIF frames
    for frame in video_clip.iter_frames(fps=frame_rate, dtype='uint8'):
        im = Image.fromarray(frame)
        im = im.quantize(colors=colors, method=method, dither=0)
        gif_frames.append(im)

    # Save the GIF using Pillow
    gif_frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=1000 / frame_rate,
        loop=0,  # 0 means loop indefinitely
        format="GIF"
    )
    # Return the processed image path
    return output_gif_path
@app.get("/get-port")
def get_server_port():
    '''
    Returns the port number of the server

    Parameters
    ----------
    None

    Returns
    -------
    str: 
        port number
    '''
    for i,arg in enumerate(sys.argv):
        if arg.startswith("--port"):
            return sys.argv[i+1]
    return '8001'

def convert_and_pack_results(name):
    '''
    Converts the .obj file to .gif and packs the results into a zip file

    Parameters
    ----------
    name: str
        Name of the .obj file

    Returns
    -------
    json: dict
        Dictionary containing the paths to the GIF and ZIP files
    '''

    # Coverting to gif
    os.system(f"python -m kiui.render logs/{name}.obj --save_video output/{name}.gif --wogui --force_cuda_rast")
    # Make the GIF loop infinitely
    # make_gif_loop_infinitely(f'output/{name}.gif', f'output/{name}.gif')

    ## Move png, mtl and obj file to a new folder name
    os.makedirs(f'logs/{name}', exist_ok=True)
    shutil.move(f'logs/{name}.obj', f'logs/{name}/{name}.obj')
    shutil.move(f'logs/{name}.mtl', f'logs/{name}/{name}.mtl')
    shutil.move(f'logs/{name}_albedo.png', f'logs/{name}/{name}_albedo.png')
    shutil.copy(f'output/{name}.gif', f'logs/{name}/{name}.gif')
    # Saving the obj, mtl and png files into a zip file
    shutil.make_archive(f'output/{name}', 'zip', f'logs/{name}')
    # Remove the logs/name folder
    shutil.rmtree(f'logs/{name}')
    
    # Clear all the files in the logs folder
    for file in os.listdir('logs'):
        ## Check if it's a file
        if os.path.isfile(f'logs/{file}'):
            ## Remove the file
            os.remove(f'logs/{file}')
        elif os.path.isdir(f'logs/{file}'):
            ## Remove the directory
            shutil.rmtree(f'logs/{file}')

    # Add gif path and zip path to a json format
    json = {"gif_path": f'output/{name}.gif', "zip_path": f'output/{name}.zip'}

    return json
## Generate Caption from Image
def generate_captions(image_path):
    '''
    Generates the caption from the image using transformers BLIP

    Parameters
    ----------
    image_path: str
        Path to the image

    Returns
    -------
    str:
        Caption generated from the image
        
    '''
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

    # Read the image
    image = Image.open(image_path).convert("RGB")

    # Unconditional image captioning 
    inputs = processor(image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)

    return text


## TO BE EDITED WITH MAGIC 3D
def get_asset_folder(userid, input_text, current_timestamp):
    unique_id = f"{userid}+{input_text}+{current_timestamp}"
    model_id = hashlib.md5(unique_id.encode()).hexdigest()
    
    return f"{OUTPUT_DIR}/{userid}/{model_id}"


def process_image(input_file: UploadFile, input_text: str, userid):
    '''
    Processes the uploaded image and the text description and converts to 3D

    Parameters
    ----------
    input_file: UploadFile  
        Uploaded image file
    input_text: str
        Text to be processed
    userid: ID
        Authenticated user unique identifier

    Returns
    -------
    json: dict
        Dictionary containing the paths to the GIF and ZIP files
    '''

    ## Remove all special characters from the save path
    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_text = input_text.lower()
    experiment_dir = input_text.replace(" ", "_")
    logs_path = "./ImageDream/outputs/imagedream-sd21-shading/" + experiment_dir
    abs_logs_path = str(Path(logs_path).resolve())
    
    asset_folder = get_asset_folder(userid, input_text, current_timestamp)
 
    ## Make a asset directory if it doesn't exist
    os.makedirs(asset_folder, exist_ok=True)
    
    # Save the uploaded image
    input_image_file_path = os.path.join(UPLOAD_DIR, input_file.filename)
    with open(input_image_file_path, "wb") as f:
        shutil.copyfileobj(input_file.file, f)

    print("The training has started..........")
    # Export PYTHONPATH
    subprocess.run('export PYTHONPATH=$PYTHONPATH:./extern/ImageDream', cwd="ImageDream/", shell=True, executable="/bin/bash")

    # Running the generation model
    subprocess.run(["python", "launch.py", "--config", image_text_to_3D_imagedream_yaml, "--train", "--gpu", "0", 
                    "name=imagedream-sd21-shading", "tag="+experiment_dir, 
                    "system.prompt_processor.prompt=" + input_text,
                    "system.prompt_processor.image_path=./data/" + input_file.filename, 
                    "system.guidance.ckpt_path=./extern/ImageDream/release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt",
                    "system.guidance.config_path=./extern/ImageDream/imagedream/configs/sd_v2_base_ipmv.yaml", "exp_root_dir=" + abs_logs_path], 
                    cwd="ImageDream/")
    # Get the path of the mp4 file
    mp4_path = glob(logs_path + "/save/*.mp4")[0]
    # Define the output GIF file path and convert the mp4 to gif
    # gif_path = os.path.join(OUTPUT_DIR, f"{directory_name}.gif")
    gif_path = os.path.join(f"{asset_folder}/{experiment_dir}.gif")
    make_gif(mp4_path, gif_path)
    print("Gif and mp4 created......") 

    # Running the export model
    subprocess.run(["python", "launch.py", "--config", image_text_to_3D_imagedream_yaml, "--export", "--gpu", "0", 
                    "resume=" + abs_logs_path + "/ckpts/last.ckpt", 
                    "system.prompt_processor.prompt=" + input_text, 
                    "system.prompt_processor.image_path=./data/" + input_file.filename,
                    "system.exporter_type=mesh-exporter", 
                    "system.geometry.isosurface_method=mc-cpu", "system.geometry.isosurface_resolution=256", ], 
                    cwd="ImageDream/")
    # Pack the .mtl, .obj model files and .jpg texture file into a single zip
    print("Export Done.....")
    zip_json = pack_results(output_path= abs_logs_path,  
                            asset_folder=asset_folder)
    zip_path  = zip_json["zip_path"]
    print("Results packed.......")

    # Remove the intermediatory files
    # deleteIntermediateFiles(path=logs_path+"/save/")
    # remove the experiment dir for the current run (all files+folders)
    delete_intermediate_files(path=abs_logs_path)
    print("Deleted intermediatory files......")

    # Return the json
    # json = {"gif_path": gif_path, "zip_path": None}
    json = {"gif_path": gif_path, "zip_path": zip_path}

    return json


# Function to process text using process_text.py
def process_text(input_text, userid):
    '''
    Processes the text and converts to 3D

    Parameters
    ----------
    input_text: str
        Text to be processed
    userid: ID
        Authenticated user unique identifier

    Returns
    -------
    json: dict
        Dictionary containing the paths to the GIF and ZIP files
    '''

    ## Remove all special characters from the save path
    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_text = input_text.lower()
    experiment_dir = input_text.replace(" ", "_")
    logs_path = "./MVDream-threestudio/outputs/mvdream-sd21-rescale0.5-shading/" + experiment_dir
    abs_logs_path = str(Path(logs_path).resolve())
    
    asset_folder = get_asset_folder(userid, input_text, current_timestamp)
    
    ## Make the asset folder directory if it doesn't exist
    os.makedirs(asset_folder, exist_ok=True)

    print("The training has started..........")
    
    # Running the generation model
    subprocess.run(["python", "launch.py", "--config", text_to_3D_shading_mvdream_yaml, "--train", "--gpu", "0", "system.prompt_processor.prompt=" + input_text, "exp_dir=" + abs_logs_path], cwd="MVDream-threestudio/")
    
    
    # Get the path of the mp4 file
    print(abs_logs_path)
    print(glob(abs_logs_path + "/save/*.mp4"))
    mp4_path = glob(abs_logs_path + "/save/*.mp4")[0]
    
    # Define the output GIF file path and convert the mp4 to gif
    # gif_path = os.path.join(OUTPUT_DIR, f"{directory_name}.gif")
    gif_path = os.path.join(f"{asset_folder}/{experiment_dir}.gif")
    make_gif(mp4_path, gif_path)
    print("Gif and mp4 created......")

    # Running the export model
    subprocess.run(["python", "launch.py", "--config", text_to_3D_shading_mvdream_yaml, "--export", "--gpu", "0", 
                    "resume=" + abs_logs_path + "/ckpts/last.ckpt", "system.exporter_type=mesh-exporter", 
                    "system.geometry.isosurface_method=mc-cpu", "system.geometry.isosurface_resolution=256", 
                    "system.prompt_processor.prompt=" + input_text], cwd="MVDream-threestudio/")

    # Pack the .mtl, .obj model files and .jpg texture file into a single zip
    print("Export Done.....")

    zip_json = pack_results(output_path= abs_logs_path,
                            asset_folder=asset_folder)
    
    zip_path  = zip_json["zip_path"]
    print("Results packed.......")

    # Remove the intermediatory files
    # deleteIntermediateFiles(path=logs_path+"/save/")
    # remove the experiment dir for the current run (all files+folders)
    delete_intermediate_files(path=abs_logs_path)
    print("Deleted intermediatory files......")

    # Return the json
    # json = {"gif_path": gif_path, "zip_path": None}
    json = {"gif_path": gif_path, "zip_path": zip_path}

    return json


def add_to_port_status(port,api):
    '''
    Adds the port number, api name and time to the csv file port_status.csv

    Parameters
    ----------
    port: str
        Port number of the server
    api: str
        Name of the api

    Returns
    -------
    None
    '''
    ## Open the csv file port_status.csv
    port_status = pd.read_csv('port_status.csv')
    ## Insert the row into the csv file with port number, api name and time
    port_status = pd.concat([port_status, pd.DataFrame([[port, api, datetime.now()]], columns=['port', 'api_name', 'time'])])
    ## Save the csv file
    port_status.to_csv('port_status.csv', index=False)

def remove_from_port_status(port):
    '''
    Removes the port number from the csv file port_status.csv

    Parameters
    ----------
    port: str
        Port number of the server

    Returns
    -------
    None
    '''

    ## Open the csv file port_status.csv
    port_status = pd.read_csv('port_status.csv')
    ## Remove the row with the port number
    port_status = port_status[port_status['port'] != port]
    ## Save the csv file
    port_status.to_csv('port_status.csv', index=False)


def get_max_steps(yaml_file_path):
    '''
    Get the max training steps from YAML file

    Parameters
    ----------
    yaml_file_path: str
        file path of the yaml file that is used for 3D generation

    Returns
    -------
    max_steps_value: str
        String containing the maximum training steps
    '''
    # Get the current working directory
    original_directory = os.getcwd()

    # Change the working directory to "MVDream"
    os.chdir('MVDream-threestudio')

    # Read the YAML file
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Get the value of the "max_steps" variable
    max_steps_value = yaml_data.get('trainer', {}).get('max_steps')

    # Change the working directory back to the original directory
    os.chdir(original_directory)

    return max_steps_value


def pack_results(output_path, asset_folder):
    '''
    packs the results into a zip file

    Parameters
    ----------
    output_path: str
        path of the output folder where all the generations are stored
    asset_folder: str
        file path where user assets from 3D generation are saved

    Returns
    -------
    json: dict
        Dictionary containing the paths to ZIP files
    '''
    yaml_file_path = f"{output_path}/configs/raw.yaml"
    max_steps = get_max_steps(yaml_file_path)

    folder_path = f"{output_path}/save/it{max_steps}-export/"
    experiment_dir = output_path.split('/')[-1]

    try:
        # Copy the gif file to the export directory
        gif_path = os.path.join(f"{asset_folder}/{experiment_dir}.gif")
        shutil.copyfile(gif_path, '{}/{}'.format(folder_path, gif_path.split('/')[-1]))
        print(f"File copied from '{gif_path}' to '{folder_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # zip_path = os.path.join(OUTPUT_DIR, f"{directory_name}")
    zip_path = os.path.join(f"{asset_folder}/{experiment_dir}")

    # Saving the texture.jpg, model.mtl and model.obj files into a zip file
    # os.makedirs(zip_path, exist_ok=True)
    shutil.make_archive(zip_path, 'zip', folder_path)
    
    # Add zip path to a json format
    json = {"zip_path": zip_path+".zip"}

    return json


def delete_intermediate_files(path: str = Form(...)):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(f"A file error occurred: {e}")


def deleteIntermediateFiles(path: str = Form(...)):
    # Get a list of all files and subdirectories in the folder
    files_and_folders = os.listdir(path)

    # Iterate through each file or folder
    for item in files_and_folders:
        item_path = os.path.join(path, item)

        # Check if it is a file with the specified extensions
        if os.path.isfile(item_path) and item.lower().endswith(('.mp4', '.png', '.jpg', '.jpeg')):
            try:
                # Delete the file
                os.remove(item_path)
            except Exception as e:
                print(f"Error deleting file {item_path}: {e}")

        # Check if it is a folder ending with "test"
        if os.path.isdir(item_path) and item.lower().endswith('test'):            
            try:
                # Delete the folder and its contents
                for root, dirs, files in os.walk(item_path, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)

                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        os.rmdir(dir_path)

                # Finally, remove the main folder
                os.rmdir(item_path)

            except Exception as e:
                print(f"Error deleting folder {item_path}: {e}")

### API ### 

@app.post("/dummy_method/")
async def dummyMethod(text:str = Form(...)):
    try:
        json = {"res": f'suffessfully', "done": f'processed'}
        return json;
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dummy method: {str(e)}")

# @app.post("/delete_intermediate_files/")
# async 
# def deleteIntermediateFiles(path: str = Form(...)):
#     # Get a list of all files in the folder
#     files = os.listdir(path)
#     print("Deleting files starting!!")
#     # Iterate through each file
#     for file in files:
#         print("Deleting each file..")
#         # Check if the file is a GIF or image (you can extend this list as needed)
#         if file.lower().endswith(('.mp4', '.png', '.jpg', '.jpeg')):
#             file_path = os.path.join(path, file)
#             try:
#                 # Delete the file
#                 os.remove(file_path)
#                 print(f"Deleted: {file_path}")
#             except Exception as e:
#                 print(f"Error deleting {file_path}: {e}")



# Export the obj files
@app.route('/download_zip/')
def download_zip(zip_file_path: str = Form(...)):
    """
    API endpoint to download a zip file for a given zip_file_path.
    """
    # return send_file(zip_file_path, as_attachment=True)
    zip_path = zip_file_path
    # Create a Path object for validation
    download_path = Path(zip_path)
    
    # Check if the file exists
    if not download_path.is_file():
        return {"error": "Download file not found"}
    
    # Return the FileResponse with the file path and name
    return FileResponse(zip_path, filename=download_path.name)


# API for generating caption
@app.post("/generate_caption/")
async def generate_caption(image: UploadFile):
    '''
    Generates the caption from the image using transformers BLIP

    Parameters
    ----------
    image: UploadFile
        Uploaded image file

    Returns
    -------
    str:
        Caption generated from the image
    '''

    # Save the uploaded image
    input_image_file_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(input_image_file_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Generate the caption
    caption = generate_captions(input_image_file_path)

    # Return the caption
    return caption

# Route to handle image uploads
@app.post("/upload-image-swagger/")
async def process_image_endpoint_swagger(image: UploadFile, text: str = Form(...), userid: str = Form(...)):
    '''
    Processes the uploaded image,text and converts to 3D to render on Swagger UI

    Parameters
    ----------
    image: UploadFile
        Uploaded image file
    text: str
        Text to be processed
    userid: ID
        Authenticated user unique identifier 

    Returns
    ------- 
    FileResponse:
        Returns the processed GIF file and renders it directly on Swagger UI
    '''
    # port = get_server_port()
    # Process the image
    try:
        # Add log to port_status.csv
        # add_to_port_status(port, 'upload-image-swagger')
        path = process_image(image,text,userid)        
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        # return FileResponse(path['gif_path'], media_type='image/gif')

        zip_path = path["zip_path"]
        # Create a Path object for validation
        download_path = Path(zip_path)
        
        # Check if the file exists
        if not download_path.is_file():
            return {"error": "Download file not found"}
        
        # Return the FileResponse with the file path and name
        return FileResponse(zip_path, media_type="application/octet-stream", filename=download_path.name)

    except Exception as e:
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    

# Route to handle text inputs
@app.post("/process-text-swagger/")
async def process_text_endpoint_swagger(text: str = Form(...), userid: str = Form(...)):
    '''
    Processes the text and converts to 3D to render on Swagger UI

    Parameters
    ----------
    text: str
        Text to be processed
    userid: ID
        Authenticated user unique identifier

    Returns
    -------
    FileResponse:
        Returns the processed GIF file and renders it directly on Swagger UI
    '''

    # Define the output GIF file path
    # port = get_server_port()
    try:
        # Add log to port_status.csv
        # add_to_port_status(port, 'process-text-swagger')
        # Process the text
        path = process_text(text,userid)
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        # Return the processed GIF
        # return FileResponse(path['gif_path'], media_type='image/gif')

        zip_path = path["zip_path"]
        # Create a Path object for validation
        download_path = Path(zip_path)
        
        # Check if the file exists
        if not download_path.is_file():
            return {"error": "Download file not found"}
        
        # Return the FileResponse with the file path and name
        return FileResponse(zip_path, media_type="application/octet-stream", filename=download_path.name)

    
    except Exception as e:
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")
    

# Route to handle text inputs and return the paths of model files as GIF and ZIP
@app.post("/process-text-highf/")
async def process_text_endpoint_json(text: str = Form(...), userid: str = Form(...)):
    '''
    Processes the text and converts to 3D

    Parameters
    ----------
    text: str
        Text to be processed
    userid: ID
        Authenticated user unique identifier

    Returns
    -------
    json: dict
        returns the dictionary containing the paths to the GIF and ZIP files
    '''

    try:
        path = process_text(text,userid)
        return path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")
    

# Route to handle image uploads and return the paths of model files as GIF and ZIP
@app.post("/upload-image-highf/")
async def process_image_endpoint_json(image: UploadFile, text: str = Form(...), userid: str = Form(...)):
    '''
    Processes the uploaded image,text and converts to 3D

    Parameters
    ----------
    image: UploadFile
        Uploaded image file
    text: str
        Text to be processed
    userid: ID
        Authenticated user unique identifier 

    Returns
    ------- 
    json: dict
        returns the dictionary containing the paths to the GIF and ZIP files
    '''
    
    try:
        path = process_image(image,text,userid)        
        return path
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")    


@app.post("/get-zip/")
async def get_zip(file_path: str = Form(...)):
    '''
    Returns the ZIP file

    Parameters
    ----------
    file_path: str
        Path to the ZIP file

    Returns
    -------
    FileResponse:
        Returns the ZIP file in octet-stream format
    '''
    
    # Getting the file name
    file_name = file_path.split('/')[-1]

    ## Check if the file extension is zip with assert
    if not file_name.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File extension is not zip")
    
    # Define the output GIF file path
    try:
        # Return the processed GIF
        return FileResponse(file_path, media_type='application/octet-stream', filename=file_name)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")


# @app.post("/render-gif-swagger/")
# async def render_gif(file_path: str = Form(...)):
#     """
#     Returns the GIF file

#     Parameters
#     ----------
#     file_path: str
#         Path to the GIF file

#     Returns
#     -------
#     FileResponse:
#         Returns the GIF file in image/gif format
#     """

#     # Getting the file name
#     file_name = file_path.split("/")[-1]

#     ## Check if the file extension is gif with assert
#     if not file_name.endswith(".gif"):
#         raise HTTPException(status_code=400, detail="File extension is not gif")

#     # Define the output GIF file path
#     try:
#         # Return the processed GIF
#         return FileResponse(file_path, media_type="image/gif")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

    
    

# @app.post("/render-gif/")
# async def render_gif(file_path: str = Form(...)):
#     '''
#     Returns the GIF file

#     Parameters
#     ----------
#     file_path: str
#         Path to the GIF file

#     Returns
#     -------
#     FileResponse:
#         Returns the GIF file in image/gif format
#     '''

#     # Getting the file name
#     file_name = file_path.split('/')[-1]

#     ## Check if the file extension is gif with assert
#     if not file_name.endswith('.gif'):
#         raise HTTPException(status_code=400, detail="File extension is not gif")

#     # Define the output GIF file path
#     try:
#         # Return the processed GIF
#         return FileResponse(file_path, media_type='image/gif', filename=file_name)
 
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

        
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)

