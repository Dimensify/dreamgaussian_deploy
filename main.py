from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import subprocess
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

# Directory to store uploaded files
UPLOAD_DIR = "./dreamgaussian/data"

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

    # Save the video using kiui.render
    # video_save_command = f"kiui.render logs/{name}.obj --save_video {name}.mp4 --wogui --force_cuda_rast"
    # subprocess.run(["python", "-m", video_save_command], shell=True)
    os.system(f"python -m kiui.render logs/{name}.obj --save_video output/{name}.gif --wogui --force_cuda_rast")

    gif_path = f'output/{name}.gif'

    # Return the gif path
    return gif_path

# Function to process text using process_text.py
def process_text(input_text):
    save_path = input_text.replace(" ", "_")
    # Replace this with the actual command to process the text
    # For example, you can use subprocess to run your Python script
    subprocess.run(["python", "dreamgaussian/main.py", "--config", "dreamgaussian/configs/text_mv.yaml", "prompt=" + input_text, f"save_path={save_path}", "force_cuda_rast=True"])
    subprocess.run(["python", "dreamgaussian/main2.py", "--config", "dreamgaussian/configs/text_mv.yaml", "prompt=" + input_text, f"save_path={save_path}", "force_cuda_rast=True"])

    # Converting to gif
    os.system(f"python -m kiui.render logs/{save_path}.obj --save_video output/{save_path}.gif --wogui --force_cuda_rast")

    # Input and output file paths
    gif_path = f'output/{save_path}.gif'
    
    # Return the gif path
    return gif_path

# Route to handle image uploads
@app.post("/upload-image/")
async def upload_image(image: UploadFile):
    # Process the image
    try:
        gif = process_image(image)

        # Return the processed GIF
        # return FileResponse(output_file_path)
        return FileResponse(gif)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    

# Route to handle text inputs
@app.post("/process-text/")
async def process_text_endpoint(text: str = Form(...)):
    # Define the output GIF file path
    try:
        # Process the text
        gif = process_text(text)
        # Return the processed GIF
        return FileResponse(gif)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
