from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import subprocess
from mangum import Mangum
from moviepy.editor import VideoFileClip
from PIL import Image

app = FastAPI()
handler = Mangum(app)

# Directory to store uploaded files
UPLOAD_DIR = "./data"

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

def make_gif(input_path,output_path):

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
    subprocess.run(["python3", "process.py", f"data/{input_file.filename}"])
    subprocess.run(["python3", "step1.py", "--config", "configs/image.yaml", "input=" + processed_image_path, f"save_path={name}", "force_cuda_rast=True"])
    subprocess.run(["python3", "step2.py", "--config", "configs/image.yaml", "input=" + processed_image_path, f"save_path={name}", "force_cuda_rast=True"])

    # Save the video using kiui.render
    # video_save_command = f"kiui.render logs/{name}.obj --save_video {name}.mp4 --wogui --force_cuda_rast"
    # subprocess.run(["python", "-m", video_save_command], shell=True)
    os.system(f"python3 -m kiui.render logs/{name}.obj --save_video {name}.mp4 --wogui --force_cuda_rast")

    # Input and output file paths
    input_video_path = f'{name}.mp4'
    output_gif_path = f'{name}.gif'

    gif_path = make_gif(input_video_path,output_gif_path)

    # Return the gif path
    return gif_path


# Function to process text using process_text.py
def process_text(input_text, output_file_path):
    # Replace this with the actual command to process the text
    # For example, you can use subprocess to run your Python script
    subprocess.run(["python3", "process_text.py", input_text, output_file_path])

# Route to handle image uploads
@app.post("/upload-image/")
async def upload_image(image: UploadFile):
    # Process the image
    try:
        output_gif_path = process_image(image)

        # Return the processed GIF
        # return FileResponse(output_file_path)
        return FileResponse(output_gif_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    

# Route to handle text inputs
@app.post("/process-text/")
async def process_text_endpoint(text: str = Form(...)):
    # Define the output GIF file path
    output_file_path = f"{UPLOAD_DIR}/text_output.gif"

    try:
        # Process the text
        process_text(text, output_file_path)
        # Return the processed GIF
        return FileResponse(output_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
