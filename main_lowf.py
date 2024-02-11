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
    return '8000'

def create_busy_file():
    '''
    Creates a busy file to indicate that the server is busy

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    with open('_busy', 'w') as f:
        f.write('')

def remove_busy_file():
    '''
    Removes the busy file to indicate that the server is free

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    if os.path.exists('_busy'):
        os.remove('_busy')


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
    make_gif_loop_infinitely(f'output/{name}.gif', f'output/{name}.gif')

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

def process_textimage(input_file: UploadFile, input_text: str):
    '''
    Processes the uploaded image and converts to 3D

    Parameters
    ----------
    input_file: UploadFile  
        Uploaded image file
    
    input_text: str
        Text to be processed

    Returns
    -------
    json: dict
        Dictionary containing the paths to the GIF and ZIP files
    '''

    # Define the output file name without extension
    name = os.path.splitext(input_file.filename)[0]
    
    # Save the uploaded image
    input_file_path = os.path.join(UPLOAD_DIR, input_file.filename)
    with open(input_file_path, "wb") as f:
        shutil.copyfileobj(input_file.file, f)

    # Define the processed image file path
    processed_image_path = os.path.join(UPLOAD_DIR, f"{name}_rgba.png")

    # Call the Python scripts using subprocess
    subprocess.run(["python", "dreamgaussian/process.py", f"dreamgaussian/data/{input_file.filename}"])
    subprocess.run(["python", "dreamgaussian/main.py", "--config", "dreamgaussian/configs/imagedream.yaml", "input=" + processed_image_path, "prompt=" + input_text, f"save_path={name}", "force_cuda_rast=True"])
    subprocess.run(["python", "dreamgaussian/main2.py", "--config", "dreamgaussian/configs/imagedream.yaml", "input=" + processed_image_path, "prompt=" + input_text, f"save_path={name}", "force_cuda_rast=True"])

    # Return the json
    return convert_and_pack_results(name)


def process_image(input_file: UploadFile):
    '''
    Processes the uploaded image and converts to 3D

    Parameters
    ----------
    input_file: UploadFile  
        Uploaded image file

    Returns
    -------
    json: dict
        Dictionary containing the paths to the GIF and ZIP files
    '''
    # Remove - and spaces from the file name
    input_file.filename = input_file.filename.replace(' ', '_').replace('-', '_')
    ## Add date time as a suffix to the file name
    input_file.filename = input_file.filename.split('.')[0] + '_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.' + input_file.filename.split('.')[1]
    # Define the output file name without extension
    name = os.path.splitext(input_file.filename)[0]
    
    ## Add date time as a suffix to the file name
    # Save the uploaded image
    input_file_path = os.path.join(UPLOAD_DIR, input_file.filename)
    with open(input_file_path, "wb") as f:
        shutil.copyfileobj(input_file.file, f)

    # Define the processed image file path
    processed_image_path = os.path.join(UPLOAD_DIR, f"{name}_rgba.png")

    # Call the Python scripts using subprocess
    subprocess.run(["python", "dreamgaussian/process.py", f"dreamgaussian/data/{input_file.filename}"])
    subprocess.run(["python", "dreamgaussian/main.py", "--config", "dreamgaussian/configs/image_sai.yaml", "input=" + processed_image_path, f"save_path={name}", "force_cuda_rast=True"])
    subprocess.run(["python", "dreamgaussian/main2.py", "--config", "dreamgaussian/configs/image_sai.yaml", "input=" + processed_image_path, f"save_path={name}", "force_cuda_rast=True"])

    # Return the json
    return convert_and_pack_results(name)

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
    save_path = "".join(e for e in input_text if e.isalnum()).lower()
    ## Add date time as a suffix to the file name
    save_path = save_path + '_' + datetime.now().strftime("%Y%m%d%H%M%S")
    # Replace this with the actual command to process the text
    # For example, you can use subprocess to run your Python script
    subprocess.run(["python", "dreamgaussian/main.py", "--config", "dreamgaussian/configs/text_mv.yaml", "prompt=" + input_text, f"save_path={save_path}", "force_cuda_rast=True"])
    subprocess.run(["python", "dreamgaussian/main2.py", "--config", "dreamgaussian/configs/text_mv.yaml", "prompt=" + input_text, f"save_path={save_path}", "force_cuda_rast=True"])

    # Return the json
    return convert_and_pack_results(save_path)

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

### API ### 

@app.post("/dummy_method/")
async def dummyMethod(text:str = Form(...)):
    try:
        json = {"res": f'suffessfully', "done": f'processed'}
        return json;
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dummy method: {str(e)}")


# Route to handle image uploads
@app.post("/upload-image-swagger/")
async def process_image_endpoint_swagger(image: UploadFile):
    '''
    Processes the uploaded image and converts to 3D to render on Swagger UI

    Parameters
    ----------
    image: UploadFile
        Uploaded image file

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
        create_busy_file()
        path = process_image(image)        
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
        return FileResponse(path['gif_path'], media_type='image/gif')

    except Exception as e:
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    

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
    # port = get_server_port()
    try:
        # Add log to port_status.csv
        # add_to_port_status(port, 'process-text-swagger')
        create_busy_file()
        # Process the text
        path = process_text(text)
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
        # Return the processed GIF
        return FileResponse(path['gif_path'], media_type='image/gif')
    
    except Exception as e:
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")
    
@app.post("/upload-image-text-swagger/")
async def process_image_text_endpoint_swagger(image: UploadFile, text: str = Form(...)):
    '''
    Processes the uploaded image and converts to 3D to render on Swagger UI

    Parameters
    ----------
    image: UploadFile
        Uploaded image file

    text: str
        Text to be processed

    Returns
    -------
    FileResponse:
        Returns the processed GIF file and renders it directly on Swagger UI
    '''
    # port = get_server_port()
    try:
        # Add log to port_status.csv
        # add_to_port_status(port, 'upload-image-text-swagger')
        create_busy_file()
        # Process the image
        path = process_textimage(image, text)
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
        # Return the processed GIF
        return FileResponse(path['gif_path'], media_type='image/gif')
    
    except Exception as e:
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    
@app.post("/upload-image-lowf/")
async def process_image_endpoint_json(image: UploadFile):
    '''
    Processes the uploaded image and converts to 3D; returns the paths to the GIF and ZIP files in json format

    Parameters
    ----------
    image: UploadFile
        Uploaded image file

    Returns
    -------
    json: dict
        Dictionary containing the paths to the GIF and ZIP files
    '''
    # port = get_server_port()
    try:
        # Add log to port_status.csv
        # add_to_port_status(port, 'upload-image-json')
        create_busy_file()
        # Process the image
        path = process_image(image)
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
        # Return the file paths in json format
        return path

    except Exception as e:
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

# Route to handle text inputs
@app.post("/process-text-lowf/")
async def process_text_endpoint_json(text: str = Form(...), userid: str = Form(...)):
    print("user_id",userid)
    '''
    Processes the text and converts to 3D; returns the paths to the GIF and ZIP files in json format

    Parameters
    ----------
    text: str
        Text to be processed

    Returns
    -------
    json: dict
        Dictionary containing the paths to the GIF and ZIP files
    '''
    # port = get_server_port()
    try:
        # Add log to port_status.csv
        # add_to_port_status(port, 'process-text-json')
        create_busy_file()
        # Process the text
        path = process_text(text)
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
        # Return the file paths in json format
        return path
    
    except Exception as e:
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

@app.post("/upload-image-text-lowf/")
async def process_image_text_endpoint_json(image: UploadFile, text: str = Form(...)):
    '''
    Processes the uploaded image and converts to 3D; returns the paths to the GIF and ZIP files in json format

    Parameters
    ----------
    image: UploadFile
        Uploaded image file

    text: str
        Text to be processed

    Returns
    -------
    json: dict
        Dictionary containing the paths to the GIF and ZIP files
    '''
    # port = get_server_port()
    try:
        # Add log to port_status.csv
        # add_to_port_status(port, 'upload-image-text-json')
        create_busy_file()
        # Process the image
        path = process_textimage(image, text)
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
        # Return the file paths in json format
        return path
    
    except Exception as e:
        # Remove log from port_status.csv
        # remove_from_port_status(port)
        remove_busy_file()
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


@app.post("/render-gif-swagger/")
async def render_gif(file_path: str = Form(...)):
    """
    Returns the GIF file

    Parameters
    ----------
    file_path: str
        Path to the GIF file

    Returns
    -------
    FileResponse:
        Returns the GIF file in image/gif format
    """

    # Getting the file name
    file_name = file_path.split("/")[-1]

    ## Check if the file extension is gif with assert
    if not file_name.endswith(".gif"):
        raise HTTPException(status_code=400, detail="File extension is not gif")

    # Define the output GIF file path
    try:
        # Return the processed GIF
        return FileResponse(file_path, media_type="image/gif")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

    
    

@app.post("/render-gif/")
async def render_gif(file_path: str = Form(...)):
    '''
    Returns the GIF file

    Parameters
    ----------
    file_path: str
        Path to the GIF file

    Returns
    -------
    FileResponse:
        Returns the GIF file in image/gif format
    '''

    # Getting the file name
    file_name = file_path.split('/')[-1]

    ## Check if the file extension is gif with assert
    if not file_name.endswith('.gif'):
        raise HTTPException(status_code=400, detail="File extension is not gif")

    # Define the output GIF file path
    try:
        # Return the processed GIF
        return FileResponse(file_path, media_type='image/gif', filename=file_name)
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
