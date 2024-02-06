from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import subprocess
from mangum import Mangum
import uvicorn

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

def port_info():
    '''
    Get the ports that are being used for API calls

    Parameters
    ----------
    None

    Returns
    -------
    list: list
        List containing the ports that are being used
    '''
    lsof = subprocess.check_output(('lsof', '-i'))
    ## Get all the lines that start with python
    lsof = [i for i in lsof.decode('utf-8').split('\n') if 'python' in i]
    ## Get (LISTEN) or (ESTABLISHED) ports along with what is running on them
    lsof = [i.split() for i in lsof if 'ESTABLISHED' in i]
    ## Only keep ports, LISTEN or ESTABLISHED, and the process running on them
    lsof = [[i[8].split('->')[0].split(':')[-1], i[-1], (i[8].split('->')[-1])] for i in lsof]
    ## Convert to dictionary
    lsof = [{'port': i[0], 'status': i[1], 'process': i[2]} for i in lsof]
    return lsof

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
        occupied_status = ['occupied' if i>0.05 else 'unoccupied' for i in occupied]
        return {"percentage": occupied, "status": occupied_status}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get occupied status: {str(e)}")
    
@app.post("/get_port_info/")
async def get_port_info():
    '''
    Returns the ports that are being used

    Parameters
    ----------
    None

    Returns
    -------
    list: list
        List containing the ports that are being used
    '''
    try:
        return port_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get port info: {str(e)}")

if __name__ == '__main__':
    print("Starting API")
    print(port_info())