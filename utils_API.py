from fastapi import FastAPI, UploadFile, Form, File, HTTPException
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
import tqdm

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

PROGRESS_DIR = ""

### Utilities Functions ###

def get_gpu_occupied():
    '''
    Get the GPU memory occupied

    Parameters
    ----------
    None

    Returns
    -------
    list: list
        List containing the GPU memory occupied percentages
    '''
    ## Getting Used Memory
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]

    ## Getting Total Memory
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_values = [int(x.split()[0]) for i, x in enumerate(memory_info)]

    return [i/j for i, j in zip(memory_used_values, memory_values)]

def get_progress(directory, total_steps=10000):
    pass

### API ###
@app.post("/get_occupied_status/")
async def get_occupied_status():
    '''
    Returns the GPU memory occupied

    Parameters
    ----------
    None

    Returns
    -------
    list: list
        List containing boolean if GPU is occupied
    '''
    try:
        occupied = get_gpu_occupied()
        occupied_status = ['occupied' if i>0.3 else 'unoccupied' for i in occupied]
        return {"percentage": occupied, "status": occupied_status}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get occupied status: {str(e)}")