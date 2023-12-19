from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import subprocess
from mangum import Mangum
from PIL import Image, ImageSequence
import pandas as pd
from datetime import datetime
import uvicorn
import sys 
from moviepy.editor import VideoFileClip
from glob import glob
from pathlib import Path

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
UPLOAD_DIR = "./dreamgaussian/data"

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

## Creating the output directory
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    target_width = 320
    target_height = 280
    video_clip = video_clip.resize((target_width, target_height))

    # Create a list to store GIF frames
    gif_frames = []

    # Generate GIF frames
    for frame in video_clip.iter_frames(fps=frame_rate, dtype='uint8'):
        gif_frames.append(Image.fromarray(frame))

    # Save the GIF using Pillow
    gif_frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=1000 / frame_rate,
        loop=0  # 0 means loop indefinitely
    )

    # Return the processed image path
    return output_gif_path

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
    return '8000'

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

## TO BE EDITED WITH MAGIC 3D

# def process_image(input_file: UploadFile):
#     '''
#     Processes the uploaded image and converts to 3D

#     Parameters
#     ----------
#     input_file: UploadFile  
#         Uploaded image file

#     Returns
#     -------
#     json: dict
#         Dictionary containing the paths to the GIF and ZIP files
#     '''

#     # Define the output file name without extension
#     name = os.path.splitext(input_file.filename)[0]
    
#     # Save the uploaded image
#     input_file_path = os.path.join(UPLOAD_DIR, input_file.filename)
#     with open(input_file_path, "wb") as f:
#         shutil.copyfileobj(input_file.file, f)

#     # Define the processed image file path
#     processed_image_path = os.path.join(UPLOAD_DIR, f"{name}_rgba.png")

#     # Call the Python scripts using subprocess
#     subprocess.run(["python", "dreamgaussian/process.py", f"dreamgaussian/data/{input_file.filename}"])
#     subprocess.run(["python", "dreamgaussian/main.py", "--config", "dreamgaussian/configs/image.yaml", "input=" + processed_image_path, f"save_path={name}", "force_cuda_rast=True"])
#     subprocess.run(["python", "dreamgaussian/main2.py", "--config", "dreamgaussian/configs/image.yaml", "input=" + processed_image_path, f"save_path={name}", "force_cuda_rast=True"])

#     # Return the json
#     return convert_and_pack_results(name)

# Function to process text using process_text.py
def process_text(input_text):
    '''
    Processes the text and converts to 3D

    Parameters
    ----------
    input_text: str
        Text to be processed

    Returns
    -------
    json: dict
        Dictionary containing the paths to the GIF and ZIP files
    '''

    ## Remove all special characters from the save path
    directory_name = input_text.replace(" ", "_")
    logs_path = "MVDream-threestudio/outputs/mvdream-sd21-rescale0.5-shading/" + directory_name

    # print("The training has started..........")
    # # Running the generation model
    # subprocess.run(["python", "launch.py", "--config", "../configs/mvdream-sd21-shading.yaml", "--train", "--gpu", "0", "system.prompt_processor.prompt=" + input_text], cwd="MVDream-threestudio/")
    # # Get the path of the mp4 file
    # mp4_path = glob(logs_path + "/save/*.mp4")[0]
    # # Define the output GIF file path and convert the mp4 to gif
    gif_path = os.path.join(OUTPUT_DIR, f"{directory_name}.gif")
    # make_gif(mp4_path, gif_path)
    # print("Gif and mp4 created......")

    # Running the export model
    subprocess.run(["python", "launch.py", "--config", "../configs/mvdream-sd21-shading.yaml", "--export", "--gpu", "0", 
                    "resume=" + "outputs/mvdream-sd21-rescale0.5-shading/" + directory_name + "/ckpts/last.ckpt", "system.exporter_type=mesh-exporter", 
                    "system.geometry.isosurface_method=mc-cpu", "system.geometry.isosurface_resolution=256", 
                    "system.prompt_processor.prompt=" + input_text], cwd="MVDream-threestudio/")

    print("Export Done.....")
    zip_json = pack_results(input_text)
    zip_path  = zip_json["zip_path"]
    print("Results packed.......")

    ## Remove the logs folder
    # shutil.rmtree(logs_path)
    deleteIntermediateFiles(path=logs_path+"/save/")
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

def pack_results(input_text):
    '''
    packs the results into a zip file

    Parameters
    ----------
    input_text: str
        input prompt text

    Returns
    -------
    json: dict
        Dictionary containing the paths to ZIP files
    '''
    ## Remove all special characters from the save path
    directory_name = input_text.replace(" ", "_")
    folder_path = f"MVDream-threestudio/outputs/mvdream-sd21-rescale0.5-shading/{directory_name}/save/it500-export/" 
    # zip_path = f'{folder_path}/results'
    # zip_path = f'{folder_path}{directory_name}'
    zip_path = os.path.join(OUTPUT_DIR, f"{directory_name}")

    # Saving the texture.jpg, model.mtl and model.obj files into a zip file
    # os.makedirs(zip_path, exist_ok=True)
    shutil.make_archive(zip_path, 'zip', folder_path)
    
    # Add zip path to a json format
    json = {"zip_path": zip_path+".zip"}

    return json


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
def deleteIntermediateFiles(path: str = Form(...)):
    # Get a list of all files in the folder
    files = os.listdir(path)
    print("Deleting files starting!!")
    # Iterate through each file
    for file in files:
        print("Deleting each file..")
        # Check if the file is a GIF or image (you can extend this list as needed)
        if file.lower().endswith(('.mp4', '.png', '.jpg', '.jpeg')):
            file_path = os.path.join(path, file)
            try:
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

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



# Route to handle image uploads
# @app.post("/upload-image-swagger/")
# async def process_image_endpoint_swagger(image: UploadFile):
#     '''
#     Processes the uploaded image and converts to 3D to render on Swagger UI

#     Parameters
#     ----------
#     image: UploadFile
#         Uploaded image file

#     Returns
#     ------- 
#     FileResponse:
#         Returns the processed GIF file and renders it directly on Swagger UI
#     '''
#     port = get_server_port()
#     # Process the image
#     try:
#         # Add log to port_status.csv
#         add_to_port_status(port, 'upload-image-swagger')
#         path = process_image(image)        
#         # Remove log from port_status.csv
#         remove_from_port_status(port)
#         return FileResponse(path['gif_path'], media_type='image/gif')

#     except Exception as e:
#         # Remove log from port_status.csv
#         remove_from_port_status(port)
#         raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    

# Route to handle text inputs
@app.post("/process-text-swagger/")
async def process_text_endpoint_swagger(text: str = Form(...)):
    '''
    Processes the text and converts to 3D to render on Swagger UI

    Parameters
    ----------
    text: str
        Text to be processed

    Returns
    -------
    FileResponse:
        Returns the processed GIF file and renders it directly on Swagger UI
    '''

    # Define the output GIF file path
    port = get_server_port()
    try:
        # Add log to port_status.csv
        add_to_port_status(port, 'process-text-swagger')
        # Process the text
        path = process_text(text)
        # Remove log from port_status.csv
        remove_from_port_status(port)
        # Return the processed GIF
        # return FileResponse(path['gif_path'], media_type='image/gif')

        zip_path = path["zip_path"]
        print("The zip path is...: ",zip_path)
        # Get the absolute path
        absolute_path = os.path.abspath(zip_path)
        print("THe full path is: ", absolute_path)
        # Create a Path object for validation
        download_path = Path(zip_path)
        
        # Check if the file exists
        if not download_path.is_file():
            return {"error": "Download file not found"}
        print("The zip path is: ",zip_path)
        # Return the FileResponse with the file path and name
        return FileResponse(zip_path, media_type="application/octet-stream", filename=download_path.name)

    
    except Exception as e:
        # Remove log from port_status.csv
        remove_from_port_status(port)
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")
    
# @app.post("/upload-image-json/")
# async def process_image_endpoint_json(image: UploadFile):
#     '''
#     Processes the uploaded image and converts to 3D; returns the paths to the GIF and ZIP files in json format

#     Parameters
#     ----------
#     image: UploadFile
#         Uploaded image file

#     Returns
#     -------
#     json: dict
#         Dictionary containing the paths to the GIF and ZIP files
#     '''
#     port = get_server_port()
#     try:
#         # Add log to port_status.csv
#         add_to_port_status(port, 'upload-image-json')
#         # Process the image
#         path = process_image(image)
#         # Remove log from port_status.csv
#         remove_from_port_status(port)
#         # Return the file paths in json format
#         return path

#     except Exception as e:
#         # Remove log from port_status.csv
#         remove_from_port_status(port)
#         raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

# # Route to handle text inputs
# @app.post("/process-text-json/")
# async def process_text_endpoint_json(text: str = Form(...)):
#     '''
#     Processes the text and converts to 3D; returns the paths to the GIF and ZIP files in json format

#     Parameters
#     ----------
#     text: str
#         Text to be processed

#     Returns
#     -------
#     json: dict
#         Dictionary containing the paths to the GIF and ZIP files
#     '''
#     port = get_server_port()
#     try:
#         # Add log to port_status.csv
#         add_to_port_status(port, 'process-text-json')
#         # Process the text
#         path = process_text(text)
#         # Remove log from port_status.csv
#         remove_from_port_status(port)
#         # Return the file paths in json format
#         return path
    
#     except Exception as e:
#         # Remove log from port_status.csv
#         remove_from_port_status(port)
#         raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

# @app.post("/get-zip/")
# async def get_zip(file_path: str = Form(...)):
#     '''
#     Returns the ZIP file

#     Parameters
#     ----------
#     file_path: str
#         Path to the ZIP file

#     Returns
#     -------
#     FileResponse:
#         Returns the ZIP file in octet-stream format
#     '''
    
#     # Getting the file name
#     file_name = file_path.split('/')[-1]

#     ## Check if the file extension is zip with assert
#     if not file_name.endswith('.zip'):
#         raise HTTPException(status_code=400, detail="File extension is not zip")
    
#     # Define the output GIF file path
#     try:
#         # Return the processed GIF
#         return FileResponse(file_path, media_type='application/octet-stream', filename=file_name)
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")


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
