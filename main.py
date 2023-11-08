from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import subprocess
from mangum import Mangum
from PIL import Image, ImageSequence
import pandas as pd
from datetime import datetime
import uvicorn

app = FastAPI()
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

def make_gif_loop_infinitely(input_gif_path, output_gif_path):
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
    ## Converting to gif
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
        os.remove(f'logs/{file}')

    # Add gif path and zip path to a json format
    json = {"gif_path": f'output/{name}.gif', "zip_path": f'output/{name}.zip'}

    return json


def process_image(input_file: UploadFile):
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
    subprocess.run(["python", "dreamgaussian/main.py", "--config", "dreamgaussian/configs/image.yaml", "input=" + processed_image_path, f"save_path={name}", "force_cuda_rast=True"])
    subprocess.run(["python", "dreamgaussian/main2.py", "--config", "dreamgaussian/configs/image.yaml", "input=" + processed_image_path, f"save_path={name}", "force_cuda_rast=True"])

    # Return the json
    return convert_and_pack_results(name)

# Function to process text using process_text.py
def process_text(input_text):
    ## Remove all special characters from the save path
    save_path = "".join(e for e in input_text if e.isalnum()).lower()
    # Replace this with the actual command to process the text
    # For example, you can use subprocess to run your Python script
    subprocess.run(["python", "dreamgaussian/main.py", "--config", "dreamgaussian/configs/text_mv.yaml", "prompt=" + input_text, f"save_path={save_path}", "force_cuda_rast=True"])
    subprocess.run(["python", "dreamgaussian/main2.py", "--config", "dreamgaussian/configs/text_mv.yaml", "prompt=" + input_text, f"save_path={save_path}", "force_cuda_rast=True"])

    # Return the json
    return convert_and_pack_results(save_path)

def add_to_port_status(port,api):
    ## Open the csv file port_status.csv
    port_status = pd.read_csv('port_status.csv')
    ## Insert the row into the csv file with port number, api name and time
    port_status = port_status.append({'port': port, 'api_name': api, 'time': datetime.now()}, ignore_index=True)
    ## Save the csv file
    port_status.to_csv('port_status.csv', index=False)

def remove_from_port_status(port):
    ## Open the csv file port_status.csv
    port_status = pd.read_csv('port_status.csv')
    ## Remove the row with the port number
    port_status = port_status[port_status['port'] != port]
    ## Save the csv file
    port_status.to_csv('port_status.csv', index=False)

### API ### 


# Route to handle image uploads
@app.post("/upload-image-swagger/")
async def process_image_endpoint_swagger(image: UploadFile):
    # Process the image
    try:
        # Add log to port_status.csv
        add_to_port_status(config.port, 'upload-image-swagger')
        path = process_image(image)        
        # Remove log from port_status.csv
        remove_from_port_status(config.port)
        return FileResponse(path['gif_path'], media_type='image/gif')

    except Exception as e:
        # Remove log from port_status.csv
        remove_from_port_status(config.port)
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    

# Route to handle text inputs
@app.post("/process-text-swagger/")
async def process_text_endpoint_swagger(text: str = Form(...)):
    # Define the output GIF file path
    try:
        # Add log to port_status.csv
        add_to_port_status(config.port, 'process-text-swagger')
        # Process the text
        path = process_text(text)
        # Remove log from port_status.csv
        remove_from_port_status(config.port)
        # Return the processed GIF
        return FileResponse(path['gif_path'], media_type='image/gif')
    
    except Exception as e:
        # Remove log from port_status.csv
        remove_from_port_status(config.port)
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")
    
@app.post("/upload-image-json/")
async def process_image_endpoint_json(image: UploadFile):
    # Process the image
    try:
        path = process_image(image)

        # Return the processed GIF
        # return FileResponse(output_file_path)
        return path

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    

# Route to handle text inputs
@app.post("/process-text-json/")
async def process_text_endpoint_json(text: str = Form(...)):
    # Define the output GIF file path
    try:
        # Process the text
        path = process_text(text)
        # Return the processed GIF
        return path
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

@app.post("/get-zip/")
async def get_zip(file_path: str = Form(...)):

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
    

@app.post("/render-gif/")
async def render_gif(file_path: str = Form(...)):

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
